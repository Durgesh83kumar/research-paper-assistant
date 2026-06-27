import time
from langchain_core.prompts import PromptTemplate
from langchain_core.output_parsers import StrOutputParser
from src.llm.factory import get_llm

MAP_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""Analyze this section of a research paper and extract:
- Key findings or contributions
- Methodology mentioned
- Important results or metrics

Section:
{text}

Concise analysis:"""
)

COMBINE_PROMPT = PromptTemplate(
    input_variables=["text"],
    template="""You are an expert research paper analyst.
Based on these section analyses, write a comprehensive summary covering:

1. **Problem Statement**: What problem does this paper solve?
2. **Proposed Approach**: What is the key methodology or innovation?
3. **Key Contributions**: List 3-5 main contributions
4. **Results**: What were the main experimental results?
5. **Limitations & Future Work**: Any limitations mentioned?

Section Analyses:
{text}

Structured Summary:"""
)


def summarize_paper(documents: list) -> str:
    llm = get_llm(use_case="summary")
    parser = StrOutputParser()

    # Limit to first 10 chunks to avoid rate limits
    documents = documents[:10]

    map_chain = MAP_PROMPT | llm | parser

    chunk_summaries = []
    for i, doc in enumerate(documents):
        summary = map_chain.invoke({"text": doc.page_content})
        chunk_summaries.append(summary)

        # Add delay every 3 chunks to avoid TPM rate limit
        if (i + 1) % 3 == 0:
            time.sleep(2)

    combined_text = "\n\n".join(chunk_summaries)
    combine_chain = COMBINE_PROMPT | llm | parser
    final_summary = combine_chain.invoke({"text": combined_text})

    return final_summary