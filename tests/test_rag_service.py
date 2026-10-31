import unittest
from unittest.mock import Mock, patch

from langchain_core.documents import Document

from services import rag_service


class RagServiceTests(unittest.TestCase):
    @patch("services.rag_service.load_vector_store")
    @patch("pathlib.Path.exists", return_value=True)
    def test_verified_sources_are_added_once(self, exists, load_store):
        documents = {}
        store = Mock()
        store.get.side_effect = lambda ids: {"ids": [item for item in ids if item in documents]}

        def add_documents(items, ids):
            documents.update(zip(ids, items))

        store.add_documents.side_effect = add_documents
        load_store.return_value = store

        rag_service.get_or_build_vector_store()
        rag_service.get_or_build_vector_store()

        self.assertEqual(store.add_documents.call_count, len(rag_service.VERIFIED_REFERENCES))
        amoxicillin = documents[rag_service.NHS_AMOXICILLIN_ID]
        self.assertEqual(amoxicillin.metadata["source"], rag_service.NHS_AMOXICILLIN_SOURCE)
        self.assertEqual(amoxicillin.metadata["verification_status"], "verified")
        self.assertEqual(amoxicillin.metadata["document_date"], "2025-06-02")

    def test_retrieval_requests_verified_documents_only(self):
        store = Mock()
        store.similarity_search_with_score.return_value = []
        self.assertEqual(rag_service.retrieve_or_abstain("test", store), [])
        store.similarity_search_with_score.assert_called_once_with(
            "test", k=3, filter={"verification_status": "verified"}
        )

    def test_retrieval_filters_by_verified_source(self):
        store = Mock()
        store.similarity_search_with_score.return_value = []
        rag_service.retrieve_or_abstain("test", store, source=rag_service.NHS_AMOXICILLIN_SOURCE)
        store.similarity_search_with_score.assert_called_once_with(
            "test", k=3, filter={"$and": [
                {"verification_status": "verified"}, {"source": rag_service.NHS_AMOXICILLIN_SOURCE}
            ]}
        )

    def test_prompt_disallows_diagnosis_and_treatment_selection(self):
        prompt = rag_service.build_prompt("What should I take?", [Document(page_content="example")])
        self.assertIn("Do not diagnose, select or recommend a medicine", prompt)
        self.assertIn("Treat symptoms and diseases mentioned by the user as unverified", prompt)

    @patch("services.rag_service.get_llm")
    def test_personal_treatment_questions_are_not_sent_to_model(self, get_llm):
        store = Mock()
        for question in ("What medicine should I take for a fever?",
                         "Should I take amoxicillin?",
                         "What is amoxicillin used for if I have a fever?"):
            with self.subTest(question=question):
                self.assertEqual(rag_service.answer_question(question, store), rag_service.SUPPORTED_QUESTION_MESSAGE)
        store.similarity_search_with_score.assert_not_called()
        get_llm.assert_not_called()

    @patch("services.rag_service.get_llm")
    def test_supported_general_question_can_be_answered(self, get_llm):
        store = Mock()
        document = Document(page_content="Example", metadata={"source": rag_service.NHS_AMOXICILLIN_SOURCE})
        store.similarity_search_with_score.return_value = [(document, 0.1)]
        get_llm.return_value.invoke.return_value = Mock(content="General information")
        answer = rag_service.answer_question("What is amoxicillin?", store)
        self.assertIn(rag_service.NHS_AMOXICILLIN_SOURCE, answer)

    @patch("services.rag_service.get_llm")
    def test_other_medicines_are_not_used_as_context(self, get_llm):
        store = Mock()
        unrelated = Document(page_content="Another medicine", metadata={"source": rag_service.NHS_PARACETAMOL_SOURCE})
        store.similarity_search_with_score.return_value = [(unrelated, 0.1)]
        self.assertEqual(rag_service.answer_question("What is cetirizine?", store), rag_service.ABSTAIN_MESSAGE)
        get_llm.assert_not_called()

    @patch("services.rag_service.get_llm")
    def test_each_verified_medicine_accepts_general_question(self, get_llm):
        store = Mock()
        get_llm.return_value.invoke.return_value = Mock(content="General information")
        for name, (_, document) in rag_service.VERIFIED_REFERENCES.items():
            with self.subTest(name=name):
                store.similarity_search_with_score.return_value = [(document, 0.1)]
                answer = rag_service.answer_question(f"What is {name}?", store)
                self.assertIn(document.metadata["source"], answer)


if __name__ == "__main__":
    unittest.main()