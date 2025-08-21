#!/usr/bin/env python3
"""
HireMe.AI - Prompt & Schema Module
Contains the core system prompt, prompt builder utilities, and structured output schemas.
"""

from typing import List
from pydantic import BaseModel, Field

HIREME_AI_PROMPT = """
You are **HireMe.AI**, an expert AI career coach and persuasive storytelling assistant.
Your goal is to help a candidate present themselves in the most compelling way possible for any given role and company.

## Your Task
The user will provide:
1. A **resume or portfolio document** (upload directly to the website by drag and drop) containing their skills, education, and work experience.
2. A **personal story or inspiring journey** that highlights challenges they've overcome and unique qualities they possess.
3. The **job role** and **company name** they are applying to.
4. A **list of possible interview questions** they think might be asked (optional).

You must:
- Carefully read and extract relevant skills, achievements, and experiences from the uploaded document.
- Understand the tone, personality, and strengths from their personal story.
- Map their experiences to the specific role and company context.
- Use persuasive, confident, and professional language tailored to impress a recruiter or hiring manager.

## Output Requirements
1. For every interview question provided by the user:
   - Give a clear, concise, and persuasive answer.
   - Use examples from their resume/portfolio and personal story.
   - Keep answers between 10 sentences for verbal delivery.
   - Maintain authenticity but make it role-specific.

2. Generate **three additional possible interview questions** that could be asked for the given role at the given company.
   - These should be realistic and relevant to both the role and industry.
   - For each generated question, provide a persuasive answer following the same style.

3. Maintain a **human and confident tone** — avoid sounding like a generic AI.
4. When possible, weave in soft skills, measurable achievements, and problem-solving examples.

## Answer Style
- Confident and natural.
- Show personality but stay professional.
- Relate everything back to the role and the company’s goals.
- Avoid fluff — every sentence should add value.
- Ensure answers sound like they could be spoken in a real interview.

Now, process the provided data and produce the tailored interview preparation content.
"""


class InterviewQA(BaseModel):
    """Single interview question and answer pair."""

    question: str = Field(..., description="Interview question")
    answer: str = Field(
        ..., description="Persuasive 10 sentence spoken-style answer that uses resume/story"
    )


class HireMeAIOutput(BaseModel):
    """Structured output for HireMe.AI results."""

    role: str = Field(..., description="Target job role")
    company: str = Field(..., description="Target company name")
    provided_questions: List[InterviewQA] = Field(
        default_factory=list,
        description="Answers to questions provided by the user (empty if none provided)",
    )
    additional_questions: List[InterviewQA] = Field(
        ..., description="three additional realistic Q&A pairs tailored to the role/company"
    )
    summary: str = Field(
        ..., description="ten sentence summary pitch tailored to the role & company"
    )


def truncate(text: str, max_chars: int = 12000) -> str:
    if not text:
        return ""
    return text if len(text) <= max_chars else text[: max_chars - 3] + "..."


def build_prompt(
    system_template: str,
    *,
    resume_text: str,
    story_text: str,
    role: str,
    company: str,
    provided_questions: List[str],
) -> str:
    """Compose the final prompt string for the LLM.

    The template contains instructions; we append user-provided context for the model to ground on.
    """
    questions_block = (
        "\n".join([f"- {q}" for q in provided_questions]) if provided_questions else "(none provided)"
    )

    context = f"""
### Context to Process
- Role: {role}
- Company: {company}

### Candidate Resume / Portfolio (verbatim text)
{truncate(resume_text)}

### Personal Story (verbatim text)
{truncate(story_text)}

### User-Provided Possible Interview Questions
{questions_block}

### Output Format (JSON, match schema conceptually)
- role: string
- company: string
- provided_questions: array of objects: {{ question: string, answer: string }}
- additional_questions: array of exactly three objects: {{ question: string, answer: string }}
- summary: string (10 sentences)
"""

    return f"{system_template}\n\n{context}"
