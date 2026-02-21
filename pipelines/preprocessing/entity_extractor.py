"""
- Take raw medical text
- Use an LLM (via LangChain) to extract structured entities
- Return: { disease, symptoms, causes, treatments }

Dataset used: MedQuAD / ChatDoctor (HuggingFace)
"""

import sys
import json
import logging
from typing import Optional
from langchain_ollama import OllamaLLM
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from datasets import load_dataset

sys.path.append(".")
from pipelines.schemas.medicalentities import MedicalEntities

logger = logging.getLogger(__name__)

EXTRACTION_PROMPT = PromptTemplate(
        input_variables=["text"],
        partial_variables={},
        template="""
            You are a medical information extraction system.
            Extract structured medical entities from the following text.

            Text:
            {text}

            {format_instructions}

            Rules:
            - Extract ONLY information explicitly mentioned in the text
            - If a field is not mentioned, return an empty list
            - severity must be: low, medium, or high
            - Return ONLY valid JSON, no explanation
        """
    )

class EntityExtractor:
    """Extract medical entity (structed data) from document (unstructed data) using LLM.
        Use Ollama locally (llama3 or mistral model).
    """

    def __init__(self, model: str = "llama3"):
        self.parser = PydanticOutputParser(pydantic_object=MedicalEntities)
        self.llm = OllamaLLM(model=model, temperature=0)
        self.prompt = EXTRACTION_PROMPT.partial(
            format_instructions=self.parser.get_format_instructions()
        )
        # define chain with langchain
        self.chain = self.prompt | self.llm | self.parser

    def extract(self, text: str) -> Optional[MedicalEntities]:
        """Extracts medical entities from a text.
           Returns None if extraction fails.

        Args:
            text (str): input text

        Returns:
            Optional[MedicalEntities]: medical entity object
        """

        try:
            result = self.chain.invoke({"text": text})
            logger.info(
                "entities_extracted",
                extra={"disease": result.disease, "n_symptoms": len(result.symptoms)}
            )
            return result
        except Exception as e:
            logger.error(f"Extraction failed: {e}")
            return None
        
    def extract_batch(self, texts: list[str]) -> list[Optional[MedicalEntities]]:
        """Batch extraction from a list of texts.

        Args:
            texts (list[str]): input text

        Returns:
            list[Optional[MedicalEntities]]: list of medical entity object
        """
        results = []
        for i, text in enumerate(texts):
            logger.info(f"Extracting entities {i+1}/{len(texts)}")
            results.append(self.extract(text))
        return results

def load_medquad_samples(n: int = 1000) -> list[str]:
    """Load n examples from the MedQuAD dataset.
       Returns a list of medical texts.

    Args:
        n (int, optional): siez of data. Defaults to 100.

    Returns:
        list[str]: results
    """
    logger.info(f"Loading {n} MedQuAD samples from HuggingFace...")
    ds = load_dataset("lavita/ChatDoctor-HealthCareMagic-100k", split=f"train[:{n}]")

    # Combine question + answer to get the full context
    texts = [f"{row['input']} {row['output']}" for row in ds]
    logger.info(f"Loaded {len(texts)} samples")
    return texts


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)

    # Simple example without dataset
    sample_text = """
    Type 2 diabetes is a chronic condition that affects the way the body
    processes blood sugar. Symptoms include increased thirst, frequent urination,
    fatigue, and blurred vision. It is caused by insulin resistance and obesity.
    Treatment involves lifestyle changes, metformin, and blood sugar monitoring.
    """

    extractor = EntityExtractor(model="llama3.1")
    entities = extractor.extract(sample_text)

    if entities:
        print("\n✅ Extracted entities :")
        print(json.dumps(entities.model_dump(), indent=2))

        