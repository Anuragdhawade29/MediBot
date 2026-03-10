import os

import streamlit as st
from dotenv import find_dotenv, load_dotenv
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_community.vectorstores import FAISS
from langchain_core.prompts import ChatPromptTemplate
from langchain_groq import ChatGroq
from langchain_huggingface import HuggingFaceEmbeddings

load_dotenv(find_dotenv())

DB_FAISS_PATH = "vectorstore/db_faiss"
GROQ_MODEL_NAME = "llama-3.1-8b-instant"
TOP_K = 3

GREETING_INPUTS = {
    "hi",
    "hello",
    "hey",
    "hii",
    "heyy",
    "good morning",
    "good afternoon",
    "good evening",
}

SMALL_TALK_REPLIES = {
    "thanks": "You are welcome. Ask another health question anytime.",
    "thank you": "You are welcome. I am here when you are ready.",
    "who are you": "I am MediBot, your medical knowledge assistant.",
    "help": "Use Quick Checkup for symptom-based guidance or Clinical Query for any medical topic.",
    "bye": "Take care. For severe symptoms, seek medical help immediately.",
}

SYMPTOM_OPTIONS = [
    "Fever",
    "Headache",
    "Cough or cold",
    "Nausea or vomiting",
    "Body pain or fatigue",
    "Breathing difficulty",
    "Chills or sweating",
    "Loss of appetite",
]

PAST_HISTORY_OPTIONS = [
    "Hypertension (BP)",
    "Diabetes (Sugar)",
    "Asthma/COPD",
    "Heart disease",
    "Kidney disease",
    "Thyroid disorder",
]

COMMON_QUESTIONS = [
    "How is fever treated at home?",
    "What are warning signs in viral infection?",
    "When should I go to emergency for breathing issues?",
]

CLINICAL_QUERY_GROUPS = {
    "Fever & Infection": [
        "When should fever be considered dangerous?",
        "What tests are needed for persistent fever?",
        "How long should viral fever last?",
    ],
    "Respiratory Issues": [
        "What are early symptoms of pneumonia?",
        "Difference between cold, flu, and COVID?",
        "What causes shortness of breath suddenly?",
    ],
    "Heart & BP": [
        "When is high blood pressure an emergency?",
        "Can headache be caused by high BP?",
        "What are warning signs of heart attack?",
    ],
    "Digestive Health": [
        "What causes nausea and vomiting?",
        "How to treat dehydration at home?",
        "What are symptoms of food poisoning?",
    ],
}


@st.cache_resource
def get_vectorstore():
    embedding_model = HuggingFaceEmbeddings(model_name="sentence-transformers/all-MiniLM-L6-v2")
    return FAISS.load_local(DB_FAISS_PATH, embedding_model, allow_dangerous_deserialization=True)


@st.cache_resource
def get_llm(api_key: str):
    return ChatGroq(
        model=GROQ_MODEL_NAME,
        temperature=0.2,
        max_tokens=520,
        api_key=api_key,
    )


def build_prompt():
    return ChatPromptTemplate.from_template(
        """You are MediBot, a professional medical information assistant.

Use only the provided context.
Use recent chat history for continuity.
If the answer is not present in context, clearly say so.
Do not claim a guaranteed cure.

Answer in this exact structure:
1) Initial Assessment
2) Why This May Happen
3) Care and Treatment Options
4) Temporary Medicines (if urgent and doctor is not immediately available)
5) Precautions
6) Red Flags (Seek Urgent Care)
7) Suggested Follow-up Question

Formatting rules:
- Keep each section short (1-3 bullet points).
- Keep total response concise and complete.
- Do not leave the Red Flags section incomplete.

For section 4:
- Suggest only common over-the-counter options.
- Mention to verify allergies, pregnancy status, age, liver/kidney disease, and current medicines first.
- Do not suggest antibiotics or prescription-only medicines unless present in context.

Recent Chat History:
{chat_history}

Context:
{context}

Question:
{input}
"""
    )


def format_recent_history(messages, max_turns: int = 4) -> str:
    convo = [m for m in messages if m.get("role") in {"user", "assistant"}]
    if not convo:
        return "No prior conversation."
    recent = convo[-(max_turns * 2) :]
    return "\n".join(
        [("User: " if m["role"] == "user" else "Assistant: ") + m["content"] for m in recent]
    )


def is_greeting(text: str) -> bool:
    return text.strip().lower() in GREETING_INPUTS


def small_talk_reply(text: str):
    return SMALL_TALK_REPLIES.get(text.strip().lower())


def render_styles():
    st.markdown(
        """
<style>
    .stApp {
        background: radial-gradient(circle at 10% 0%, #e7f3ff 0%, #f7fbff 32%, #ffffff 100%);
    }
    .hero {
        border: 1px solid #d9e6f7;
        border-radius: 16px;
        padding: 16px 18px;
        background: linear-gradient(135deg, #edf5ff 0%, #f9fcff 100%);
        box-shadow: 0 8px 20px rgba(15, 23, 42, 0.05);
        margin-bottom: 12px;
    }
    .hero h2 {
        margin: 0;
        color: #0f172a;
        font-size: 1.55rem;
    }
    .hero p {
        margin: 8px 0 0 0;
        color: #334155;
        font-size: 0.95rem;
    }
    .mode-label {
        color: #334155;
        font-size: 0.9rem;
        font-weight: 700;
        margin-bottom: 6px;
    }
</style>
""",
        unsafe_allow_html=True,
    )


def normalize_temp_to_f(temp_value, temp_unit):
    if temp_value is None:
        return None
    if temp_unit == "C":
        return (temp_value * 9 / 5) + 32
    return temp_value


def calculate_risk_score(case_data, followup_answers):
    score = 0
    reasons = []

    fever_f = normalize_temp_to_f(case_data["temp_value"], case_data["temp_unit"])
    if fever_f is not None and fever_f > 101:
        score += 2
        reasons.append("Fever > 101F (+2)")

    if "Breathing difficulty" in case_data["symptoms"]:
        score += 3
        reasons.append("Breathing difficulty (+3)")

    if case_data["age"] is not None and case_data["age"] > 60:
        score += 2
        reasons.append("Age > 60 (+2)")

    if "Diabetes (Sugar)" in case_data["past_history"]:
        score += 2
        reasons.append("Diabetes (+2)")

    if "Heart disease" in case_data["past_history"]:
        score += 3
        reasons.append("Heart disease (+3)")

    if followup_answers.get("worse_lying_down") == "Yes":
        score += 2
        reasons.append("Breathing worsens while lying down (+2)")

    if followup_answers.get("cannot_hold_fluids") == "Yes":
        score += 2
        reasons.append("Unable to keep fluids (+2)")

    if score <= 3:
        level = "Mild"
        action = "Home care with monitoring."
    elif score <= 6:
        level = "Moderate"
        action = "Consult doctor within 24 hours."
    else:
        level = "High risk"
        action = "Seek urgent medical care now."

    return score, level, action, reasons


def generate_differential(case_data):
    symptoms = set(case_data["symptoms"])
    differentials = []

    if "Fever" in symptoms and "Cough or cold" in symptoms:
        differentials.extend(
            [
                "Viral upper respiratory infection",
                "Influenza",
                "COVID-like illness",
                "Bacterial sinus infection",
            ]
        )
    if "Breathing difficulty" in symptoms:
        differentials.extend(
            [
                "Acute bronchitis or lower respiratory infection",
                "Asthma/COPD flare",
                "Pneumonia",
                "Cardiac-related breathlessness",
            ]
        )
    if "Nausea or vomiting" in symptoms:
        differentials.extend(
            [
                "Viral gastroenteritis",
                "Food-borne illness",
                "Medication side effect",
                "Dehydration-related illness",
            ]
        )

    if not differentials:
        differentials = [
            "Viral syndrome",
            "Non-specific inflammatory condition",
            "Medication-related symptoms",
            "Condition requiring clinical examination",
        ]

    # Deduplicate while preserving order.
    unique = []
    for item in differentials:
        if item not in unique:
            unique.append(item)
    return unique[:4]


def generate_recommended_tests(case_data):
    symptoms = set(case_data["symptoms"])
    tests = []

    if "Fever" in symptoms or "Chills or sweating" in symptoms:
        tests.extend(["CBC", "CRP"])
    if "Breathing difficulty" in symptoms or "Cough or cold" in symptoms:
        tests.extend(["Pulse oximetry", "Chest X-ray"])
    if "Diabetes (Sugar)" in case_data["past_history"]:
        tests.append("Blood sugar monitoring")
    if "Nausea or vomiting" in symptoms:
        tests.append("Serum electrolytes")

    if not tests:
        tests = ["CBC", "Basic metabolic panel"]

    unique = []
    for item in tests:
        if item not in unique:
            unique.append(item)
    return unique


def get_dynamic_suggestions_from_symptoms(symptoms):
    symptom_set = set(symptoms or [])
    suggestions = []
    if "Fever" in symptom_set:
        suggestions.extend(
            [
                "When should fever become a concern?",
                "What infections commonly cause fever?",
            ]
        )
    if "Cough or cold" in symptom_set:
        suggestions.extend(
            [
                "Could this be flu or viral infection?",
                "When is cough considered serious?",
            ]
        )
    if "Breathing difficulty" in symptom_set:
        suggestions.extend(
            [
                "When should I go to emergency for breathing difficulty?",
                "What tests are used for breathlessness?",
            ]
        )
    if "Nausea or vomiting" in symptom_set:
        suggestions.extend(
            [
                "How to prevent dehydration in vomiting?",
                "When does vomiting need urgent care?",
            ]
        )
    # Deduplicate and limit.
    unique = []
    for item in suggestions:
        if item not in unique:
            unique.append(item)
    return unique[:6]


def build_followup_questions(case_data):
    questions = []
    symptoms = set(case_data["symptoms"])

    if "Breathing difficulty" in symptoms:
        questions.append(
            ("breathing_now", "Are you currently experiencing breathing difficulty right now?")
        )
        questions.append(
            ("worse_lying_down", "Is the breathing difficulty worsening while lying down?")
        )
    if "Fever" in symptoms:
        questions.append(("fever_3_days", "Has fever lasted more than 3 days?"))
    if "Nausea or vomiting" in symptoms:
        questions.append(("cannot_hold_fluids", "Are you unable to keep liquids down?"))

    return questions


def build_quick_check_prompt(case_data, followup_answers, risk_level, risk_action, differential, tests):
    symptoms_text = ", ".join(case_data["symptoms"]) if case_data["symptoms"] else "Not specified"
    history_text = ", ".join(case_data["past_history"]) if case_data["past_history"] else "Not provided"
    temp_text = (
        f"{case_data['temp_value']} {case_data['temp_unit']}" if case_data["temp_value"] is not None else "Not provided"
    )
    followup_text = (
        "; ".join([f"{k}={v}" for k, v in followup_answers.items()]) if followup_answers else "None"
    )
    diff_text = ", ".join(differential)
    test_text = ", ".join(tests)

    return (
        "QUICK CHECKUP TRIAGE REQUEST:\n"
        f"- Symptoms: {symptoms_text}\n"
        f"- Age: {case_data['age'] if case_data['age'] is not None else 'Not provided'}\n"
        f"- Temperature: {temp_text}\n"
        f"- Past history: {history_text}\n"
        f"- Ongoing medicines: {case_data['ongoing_medicines'] or 'Not provided'}\n"
        f"- Additional notes: {case_data['extra_notes'] or 'None'}\n"
        f"- Follow-up answers: {followup_text}\n"
        f"- Risk level: {risk_level}\n"
        f"- Suggested action: {risk_action}\n"
        f"- Possible conditions: {diff_text}\n"
        f"- Recommended tests: {test_text}\n\n"
        "Provide concise, practical doctor-style guidance based on this triage profile."
    )


def append_assistant_message(text: str):
    st.session_state.messages.append({"role": "assistant", "content": text})


def append_user_message(text: str):
    st.session_state.messages.append({"role": "user", "content": text})


def get_groq_api_key():
    # Normalize common .env formatting mistakes: extra spaces or surrounding quotes.
    raw = (os.environ.get("GROQ_API_KEY") or "").strip().strip('"').strip("'")
    return raw or None


def query_rag(prompt: str):
    vectorstore = get_vectorstore()
    if vectorstore is None:
        return "I could not load medical memory right now. Please try again.", []

    groq_api_key = get_groq_api_key()
    if not groq_api_key:
        return "Configuration issue: GROQ_API_KEY is missing in your environment.", []

    llm = get_llm(groq_api_key)
    combine_docs_chain = create_stuff_documents_chain(llm, build_prompt())
    rag_chain = create_retrieval_chain(
        vectorstore.as_retriever(search_kwargs={"k": TOP_K}),
        combine_docs_chain,
    )

    chat_history = format_recent_history(st.session_state.messages[:-1], max_turns=4)
    try:
        with st.spinner("Analyzing..."):
            response = rag_chain.invoke({"input": prompt, "chat_history": chat_history})
    except Exception as exc:
        err = str(exc)
        if "invalid_api_key" in err or "401" in err or "AuthenticationError" in err:
            return (
                "Authentication failed: your GROQ_API_KEY is invalid. "
                "Update it in `.env`, then restart Streamlit.",
                [],
            )
        return (
            "I ran into a temporary processing issue. Please try again.",
            [],
        )

    answer = response.get("answer", "I do not have a reliable answer from current context.")
    refs = []
    for doc in response.get("context", [])[:3]:
        source = doc.metadata.get("source", "").split("\\")[-1]
        page = doc.metadata.get("page_label", doc.metadata.get("page", "N/A"))
        refs.append(f"{source} p.{page}")
    return answer, refs


def finalize_quick_checkup():
    case_data = st.session_state.quick_case
    followup_answers = st.session_state.followup_answers

    score, level, action, reasons = calculate_risk_score(case_data, followup_answers)
    differential = generate_differential(case_data)
    tests = generate_recommended_tests(case_data)

    risk_rows = [
        "| Metric | Value |",
        "|---|---|",
        f"| Score | **{score}** |",
        f"| Risk Level | **{level}** |",
        f"| Suggested Action | **{action}** |",
    ]
    factors_line = ", ".join(reasons) if reasons else "No high-risk factors detected."
    triage_text = "\n".join(
        [
            "### Risk Score Summary",
            *risk_rows,
            f"**Contributing Factors:** {factors_line}",
            "",
            "### Possible Conditions",
            *[f"{idx}. {name}" for idx, name in enumerate(differential, start=1)],
            "",
            "### Recommended Tests",
            *[f"- {test}" for test in tests],
        ]
    )
    rag_prompt = build_quick_check_prompt(
        case_data=case_data,
        followup_answers=followup_answers,
        risk_level=level,
        risk_action=action,
        differential=differential,
        tests=tests,
    )
    answer, refs = query_rag(rag_prompt)
    st.session_state.quick_result = {
        "summary": triage_text,
        "answer": answer,
        "refs": refs,
    }

    st.session_state.followup_queue = []
    st.session_state.followup_index = 0
    st.session_state.followup_answers = {}
    st.session_state.quick_case = None


def start_quick_checkup(case_data):
    questions = build_followup_questions(case_data)
    st.session_state.quick_case = case_data
    st.session_state.last_quick_symptoms = case_data.get("symptoms", [])
    st.session_state.followup_queue = questions
    st.session_state.followup_index = 0
    st.session_state.followup_answers = {}

    if questions:
        st.session_state.quick_followup_question = questions[0][1]
    else:
        finalize_quick_checkup()


def handle_followup_ui():
    queue = st.session_state.get("followup_queue", [])
    idx = st.session_state.get("followup_index", 0)
    if not queue or idx >= len(queue):
        return False

    key, question = queue[idx]
    with st.container(border=True):
        st.markdown("**AI Follow-up Check**")
        st.write(question)
        response = st.radio(
            "Answer",
            ["Yes", "No"],
            horizontal=True,
            key=f"followup_answer_{idx}",
        )
        if st.button("Submit Follow-up", use_container_width=True):
            st.session_state.followup_answers[key] = response
            st.session_state.followup_index += 1
            next_idx = st.session_state.followup_index
            if next_idx < len(queue):
                st.session_state.quick_followup_question = queue[next_idx][1]
            else:
                finalize_quick_checkup()
            st.rerun()
    return True


def run_clinical_query(prompt: str):
    append_user_message(prompt)

    if is_greeting(prompt):
        append_assistant_message(
            "Hello, I am MediBot. I can guide you with symptoms, precautions, and treatment-oriented information."
        )
        return

    local_reply = small_talk_reply(prompt)
    if local_reply:
        append_assistant_message(local_reply)
        return

    answer, refs = query_rag(prompt)
    append_assistant_message(answer)
    if refs:
        st.caption("References: " + " | ".join(refs))


def initialize_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []
    if "bootstrapped" not in st.session_state:
        st.session_state.bootstrapped = True
        st.session_state.messages.append(
            {
                "role": "assistant",
                "content": "Welcome. Choose a mode and start your medical query.",
            }
        )
    st.session_state.setdefault("followup_queue", [])
    st.session_state.setdefault("followup_index", 0)
    st.session_state.setdefault("followup_answers", {})
    st.session_state.setdefault("quick_case", None)
    st.session_state.setdefault("last_quick_symptoms", [])
    st.session_state.setdefault("quick_result", None)
    st.session_state.setdefault("quick_followup_question", "")


def render_chat_history():
    if not st.session_state.messages:
        return
    st.divider()
    for message in st.session_state.messages:
        st.chat_message(message["role"]).markdown(message["content"])


def main():
    st.set_page_config(page_title="MediBot", page_icon=":stethoscope:", layout="centered")
    render_styles()
    st.markdown(
        """
<div class="hero">
  <h2>MediBot</h2>
  <p>Clinical clarity for every question, from symptoms to next steps.</p>
</div>
""",
        unsafe_allow_html=True,
    )

    initialize_state()
    st.markdown('<div class="mode-label">Conversation Mode</div>', unsafe_allow_html=True)
    mode = st.radio(
        "Conversation Mode",
        ["Quick Checkup", "Clinical Query"],
        horizontal=True,
        label_visibility="collapsed",
    )

    followup_active = handle_followup_ui()

    if mode == "Quick Checkup" and not followup_active:
        if st.session_state.quick_result:
            st.markdown("### Quick Checkup Result")
            st.markdown(st.session_state.quick_result["summary"])
            st.markdown("### Clinical Guidance")
            st.markdown(st.session_state.quick_result["answer"])
            if st.session_state.quick_result["refs"]:
                st.caption("References: " + " | ".join(st.session_state.quick_result["refs"]))
            st.info("Start a new quick checkup from the section below.")

        with st.expander("Start / Update Quick Checkup", expanded=not bool(st.session_state.quick_result)):
            st.markdown("Select one or more symptoms:")
            selected_symptoms = st.multiselect(
                "Symptoms",
                SYMPTOM_OPTIONS,
                label_visibility="collapsed",
                placeholder="Choose symptoms...",
            )
            age = st.number_input("Age", min_value=0, max_value=120, value=None, placeholder="Enter age (optional)")

            past_history = st.multiselect(
                "Past History",
                PAST_HISTORY_OPTIONS,
                placeholder="Select past conditions (optional)",
            )

            ongoing_medicines = st.text_input(
                "Current medicines",
                placeholder="Example: Metformin 500 mg, Amlodipine 5 mg",
            )

            c1, c2 = st.columns([2, 1])
            with c1:
                temp_value = st.number_input(
                    "Body Temperature",
                    min_value=30.0,
                    max_value=110.0,
                    value=None,
                    placeholder="Enter temperature (optional)",
                )
            with c2:
                temp_unit = st.selectbox("Unit", ["C", "F"])

            extra_notes = st.text_input(
                "Additional notes",
                placeholder="Example: Symptoms started 2 days ago; pain worsens at night",
            )

            c3, c4 = st.columns([1, 1])
            with c3:
                if st.button("Run Quick Checkup", use_container_width=True):
                    if (
                        not selected_symptoms
                        and temp_value is None
                        and age is None
                        and not extra_notes.strip()
                        and not past_history
                        and not ongoing_medicines.strip()
                    ):
                        st.warning("Select at least one symptom or add details.")
                    else:
                        case_data = {
                            "symptoms": selected_symptoms,
                            "age": age,
                            "temp_value": temp_value,
                            "temp_unit": temp_unit,
                            "past_history": past_history,
                            "ongoing_medicines": ongoing_medicines.strip(),
                            "extra_notes": extra_notes.strip(),
                        }
                        start_quick_checkup(case_data)
                        st.rerun()
            with c4:
                if st.button("Clear Chat", use_container_width=True):
                    st.session_state.messages = []
                    st.session_state.bootstrapped = False
                    st.session_state.followup_queue = []
                    st.session_state.followup_index = 0
                    st.session_state.followup_answers = {}
                    st.session_state.quick_case = None
                    st.session_state.quick_result = None
                    st.session_state.quick_followup_question = ""
                    st.rerun()

        render_chat_history()

    if mode == "Clinical Query" and not followup_active:
        st.markdown("**Popular Questions**")
        cols = st.columns(len(COMMON_QUESTIONS))
        for idx, question in enumerate(COMMON_QUESTIONS):
            if cols[idx].button(question, use_container_width=True):
                run_clinical_query(question)

        st.markdown("**Category-based Suggestions**")
        selected_category = st.selectbox("Category", list(CLINICAL_QUERY_GROUPS.keys()))
        category_questions = CLINICAL_QUERY_GROUPS[selected_category]
        cat_cols = st.columns(len(category_questions))
        for idx, question in enumerate(category_questions):
            if cat_cols[idx].button(question, key=f"cat_{selected_category}_{idx}", use_container_width=True):
                run_clinical_query(question)

        dynamic_questions = get_dynamic_suggestions_from_symptoms(st.session_state.get("last_quick_symptoms", []))
        if dynamic_questions:
            st.markdown("**Suggested from your recent symptoms**")
            dyn_cols = st.columns(min(3, len(dynamic_questions)))
            for idx, question in enumerate(dynamic_questions):
                col = dyn_cols[idx % len(dyn_cols)]
                if col.button(question, key=f"dyn_{idx}", use_container_width=True):
                    run_clinical_query(question)

        typed_prompt = st.chat_input("Type your question...")
        if typed_prompt:
            run_clinical_query(typed_prompt)
        render_chat_history()


if __name__ == "__main__":
    main()
