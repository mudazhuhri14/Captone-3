import streamlit as st
from langchain_openai import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_qdrant import QdrantVectorStore
from langchain.tools import tool
from langgraph.prebuilt import create_react_agent
from langchain_core.messages import ToolMessage

QDRANT_URL = st.secrets["QDRANT_URL"]
QDRANT_API_KEY = st.secrets["QDRANT_API_KEY"]
OPENAI_API_KEY = st.secrets["OPENAI_API_KEY"]

# model
llm = ChatOpenAI(model="gpt-4o-mini", api_key=OPENAI_API_KEY)
embeddings = OpenAIEmbeddings(model="text-embedding-3-small", api_key=OPENAI_API_KEY)

# qdrant
collection_name = "Data_IMDB"
qdrant = QdrantVectorStore.from_existing_collection(
    embedding=embeddings,
    collection_name=collection_name,
    url=QDRANT_URL,
    api_key=QDRANT_API_KEY
)

# sesuai mood
MOOD_GENRE_MAP = {
    "😊 Senang": "Comedy Animation Adventure Family",
    "😢 Sedih": "Drama Romance War",
    "🤔 Berpikir": "Biography History Mystery",
    "😱 Tegang": "Crime Thriller Horror",
    "🚀 Semangat": "Action Adventure Sci-Fi",
    "❤️ Romantis": "Romance Comedy Drama"
}

MOOD_BG = {
    "😊 Senang":   ("linear-gradient(135deg, #f7971e, #ffd200)", ["☀️","🌈","😄","🌻","🎉","✨","🎊","🌞"]),
    "😢 Sedih":    ("linear-gradient(135deg, #1e3c72, #2a5298)", ["🌧️","💧","😢","🌊","☔","💙","🌀","❄️"]),
    "🤔 Berpikir": ("linear-gradient(135deg, #373b44, #4286f4)", ["💭","🔮","🌌","⚡","🧠","📚","🔭","💡"]),
    "😱 Tegang":   ("linear-gradient(135deg, #200122, #6f0000)", ["👻","💀","😱","🕷️","🔪","🩸","🦇","⚰️"]),
    "🚀 Semangat": ("linear-gradient(135deg, #0f0c29, #302b63)", ["🚀","⚡","🌟","💥","🔥","🏆","💪","🎯"]),
    "❤️ Romantis": ("linear-gradient(135deg, #ff416c, #ff4b2b)", ["❤️","🌹","💕","💘","🌸","💗","💋","🥰"]),
}

# tool
@tool
def get_relevant_docs(question):
    """Use this tool to get relevant movie data from IMDB database based on user query."""
    results = qdrant.similarity_search(question, k=5)
    return results

tools = [get_relevant_docs]

# function
def chat_movie(question, history):
    agent = create_react_agent(
        model=llm,
        tools=tools,
        prompt='''Kamu adalah asisten rekomendasi film yang ahli dengan pengetahuan mendalam tentang film-film IMDB.
        SELALU jawab dalam Bahasa Indonesia.
        Hanya jawab pertanyaan yang berkaitan dengan film, genre, sutradara, aktor, dan rating.
        Selalu gunakan tools yang tersedia untuk mengambil detail film sebelum menjawab.
        Saat merekomendasikan film, sertakan: Judul, Genre, Rating, Sutradara, dan deskripsi singkat dalam Bahasa Indonesia.'''
    )
    result = agent.invoke(
        {"messages": [{"role": "user", "content": question}]}
    )
    answer = result["messages"][-1].content

    total_input_tokens = 0
    total_output_tokens = 0
    for message in result["messages"]:
        if "usage_metadata" in message.response_metadata:
            total_input_tokens += message.response_metadata["usage_metadata"]["input_tokens"]
            total_output_tokens += message.response_metadata["usage_metadata"]["output_tokens"]
        elif "token_usage" in message.response_metadata:
            total_input_tokens += message.response_metadata["token_usage"].get("prompt_tokens", 0)
            total_output_tokens += message.response_metadata["token_usage"].get("completion_tokens", 0)

    price = 17_000 * (total_input_tokens * 0.15 + total_output_tokens * 0.6) / 1_000_000

    tool_messages = []
    for message in result["messages"]:
        if isinstance(message, ToolMessage):
            tool_messages.append(message.content)

    return {
        "answer": answer,
        "price": price,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
        "tool_messages": tool_messages
    }

# ── UI ──────────────────────────────────────────────
current_mood = st.session_state.get("mood", None)
if current_mood and current_mood in MOOD_BG:
    bg, emoji_list = MOOD_BG[current_mood]
else:
    bg = "linear-gradient(135deg, #1a1a2e, #16213e, #0f3460)"
    emoji_list = ["🎬","🍿","⭐","🎭","🎥","🏆","🎞️","🌟"]

positions = [
    ("5%","3%"), ("10%","90%"), ("25%","8%"), ("40%","85%"),
    ("55%","5%"), ("70%","92%"), ("80%","15%"), ("90%","75%")
]

emoji_divs = ""
for i, emoji in enumerate(emoji_list[:8]):
    top, left = positions[i]
    emoji_divs += f'<div class="emoji-bg" style="top:{top};left:{left}">{emoji}</div>'

st.markdown(f"""
<style>
.stApp {{
    background: {bg};
    transition: background 1s ease;
}}
.emoji-bg {{
    position: fixed;
    font-size: 2.5rem;
    opacity: 0.12;
    z-index: 0;
    pointer-events: none;
    animation: float 6s ease-in-out infinite;
}}
@keyframes float {{
    0%, 100% {{ transform: translateY(0px); }}
    50% {{ transform: translateY(-20px); }}
}}
</style>
{emoji_divs}
""", unsafe_allow_html=True)

st.title("🎬 Movies Recommendation base on Feeling")
st.caption("✨ Powered by IMDB Database + RAG Agent | Find your perfect movie based on your mood!")

# output memilih mood
if "mood" not in st.session_state:
    st.session_state.mood = None

if st.session_state.mood is None:
    st.markdown("### 🎭 Bagaimana mood kamu hari ini?")
    cols = st.columns(3)
    moods = list(MOOD_GENRE_MAP.keys())
    for i, mood in enumerate(moods):
        with cols[i % 3]:
            if st.button(mood, use_container_width=True):
                st.session_state.mood = mood
                st.rerun()
else:
    st.success(f"Mood kamu: {st.session_state.mood}")
    if st.button("Ganti Mood"):
        st.session_state.mood = None
        st.session_state.messages = []
        st.rerun()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    for message in st.session_state.messages:
        with st.chat_message(message["role"]):
            st.markdown(message["content"])

    if len(st.session_state.messages) == 0:
        mood_genre = MOOD_GENRE_MAP[st.session_state.mood]
        auto_prompt = f"Rekomendasikan film dengan genre {mood_genre} karena mood saya sedang {st.session_state.mood}"
        with st.chat_message("AI"):
            response = chat_movie(auto_prompt, "")
            answer = response["answer"]
            st.markdown(answer)
            st.session_state.messages.append({"role": "AI", "content": answer})

    if prompt := st.chat_input("Tanya lebih lanjut tentang film..."):
        mood_genre = MOOD_GENRE_MAP[st.session_state.mood]
        enriched_prompt = f"[Mood user: {st.session_state.mood}, prefer genre: {mood_genre}] {prompt}"

        with st.chat_message("Human"):
            st.markdown(prompt)
        st.session_state.messages.append({"role": "Human", "content": prompt})

        with st.chat_message("AI"):
            response = chat_movie(enriched_prompt, "")
            answer = response["answer"]
            st.markdown(answer)
            st.session_state.messages.append({"role": "AI", "content": answer})

        with st.expander("**🔧 Tool Calls:**"):
            st.code(response["tool_messages"])

        with st.expander("**💬 History Chat:**"):
            messages_history = st.session_state.get("messages", [])[-20:]
            history = "\n".join([f'{msg["role"]}: {msg["content"]}' for msg in messages_history])
            st.code(history)

        with st.expander("**📊 Usage Details:**"):
            st.code(
                f'Input Tokens  : {response["total_input_tokens"]}\n'
                f'Output Tokens : {response["total_output_tokens"]}\n'
                f'Estimated Cost: Rp {response["price"]:.4f}'
            )