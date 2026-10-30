import os
import streamlit as st
import pandas as pd
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
import streamlit.components.v1 as components

# Import RAG retrieval chain
try:
    from rag_engine import get_rag_chain
except ImportError as e:
    get_rag_chain = None
    rag_import_error = str(e)

# ==========================================
# 1. PAGE CONFIGURATION
# ==========================================
st.set_page_config(
    page_title="Lumina Books | 3D Recommender Engine",
    page_icon="📚",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ==========================================
# 2. 3D BACKGROUND COMPONENT (HTML5 / JS)
# ==========================================
def render_3d_background():
    canvas_code = """
    <!DOCTYPE html>
    <html>
    <head>
        <style>
            body, html {
                margin: 0;
                padding: 0;
                overflow: hidden;
                background: transparent;
            }
            #canvas-3d {
                position: fixed;
                top: 0;
                left: 0;
                width: 100vw;
                height: 100vh;
                z-index: -1;
                pointer-events: none;
            }
        </style>
    </head>
    <body>
        <canvas id="canvas-3d"></canvas>
        <script>
            const canvas = document.getElementById('canvas-3d');
            const ctx = canvas.getContext('2d');
            let w, h;
            function resize() {
                w = canvas.width = window.innerWidth;
                h = canvas.height = window.innerHeight;
            }
            window.addEventListener('resize', resize);
            resize();

            const particles = [];
            const numParticles = 65;

            class Node3D {
                constructor() {
                    this.x = (Math.random() - 0.5) * w;
                    this.y = (Math.random() - 0.5) * h;
                    this.z = Math.random() * 800 + 100;
                    this.vx = (Math.random() - 0.5) * 0.4;
                    this.vy = (Math.random() - 0.5) * 0.4;
                    this.vz = (Math.random() - 0.5) * 0.5;
                }
                update() {
                    this.x += this.vx;
                    this.y += this.vy;
                    this.z += this.vz;
                    if (this.z < 50) this.z = 850;
                    if (this.z > 850) this.z = 50;
                }
                draw() {
                    const fov = 400;
                    const scale = fov / this.z;
                    const sx = this.x * scale + w / 2;
                    const sy = this.y * scale + h / 2;
                    const alpha = Math.min(1, Math.max(0.1, 1 - this.z / 900));

                    ctx.beginPath();
                    ctx.arc(sx, sy, Math.max(1, 2.5 * scale), 0, Math.PI * 2);
                    ctx.fillStyle = `rgba(168, 85, 247, ${alpha * 0.8})`;
                    ctx.shadowBlur = 12;
                    ctx.shadowColor = "#6366f1";
                    ctx.fill();
                }
            }

            for(let i = 0; i < numParticles; i++) particles.push(new Node3D());

            function animate() {
                ctx.clearRect(0, 0, w, h);
                particles.forEach(p => {
                    p.update();
                    p.draw();
                });
                requestAnimationFrame(animate);
            }
            animate();
        </script>
    </body>
    </html>
    """
    components.html(canvas_code, height=0, width=0)

# ==========================================
# 3. ADVANCED 3D CSS STYLING
# ==========================================
def inject_custom_styles():
    st.markdown("""
    <style>
        @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;600;800&family=Playfair+Display:ital,wght@0,600;1,400&display=swap');

        /* App-wide background */
        .stApp {
            background: radial-gradient(circle at 15% 15%, #180d2e 0%, #0a0614 55%, #030108 100%);
            font-family: 'Plus Jakarta Sans', sans-serif;
            color: #f1f5f9;
        }

        /* Glassmorphism Navigation & Tabs */
        .stTabs [data-baseweb="tab-list"] {
            gap: 1.5rem;
            background: rgba(255, 255, 255, 0.03);
            padding: 8px 18px;
            border-radius: 9999px;
            border: 1px solid rgba(255, 255, 255, 0.08);
            backdrop-filter: blur(16px);
            margin-bottom: 2rem;
            justify-content: center;
        }

        .stTabs [data-baseweb="tab"] {
            color: #94a3b8;
            font-weight: 600;
            border: none;
            background: transparent;
            transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
        }

        .stTabs [aria-selected="true"] {
            color: #ffffff !important;
            background: linear-gradient(135deg, rgba(168, 85, 247, 0.4), rgba(99, 102, 241, 0.4)) !important;
            border-radius: 9999px !important;
            box-shadow: 0 4px 20px rgba(147, 51, 234, 0.3);
        }

        /* 3D Interactive Book Presentation */
        .book-card-3d-wrap {
            perspective: 1200px;
            margin: 1.2rem 0;
        }

        .book-card-3d {
            position: relative;
            background: rgba(255, 255, 255, 0.03);
            border-radius: 18px;
            padding: 18px;
            border: 1px solid rgba(255, 255, 255, 0.07);
            backdrop-filter: blur(12px);
            transform-style: preserve-3d;
            transition: transform 0.4s cubic-bezier(0.175, 0.885, 0.32, 1.275), box-shadow 0.4s ease, border-color 0.4s ease;
        }

        .book-card-3d:hover {
            transform: translateY(-8px) rotateX(6deg) rotateY(-4deg) scale(1.02);
            box-shadow: 0 25px 35px -10px rgba(0, 0, 0, 0.7), 0 0 25px 2px rgba(168, 85, 247, 0.35);
            border-color: rgba(168, 85, 247, 0.5);
        }

        .book-cover-container {
            position: relative;
            width: 100%;
            height: 250px;
            border-radius: 12px;
            overflow: hidden;
            box-shadow: 6px 10px 20px rgba(0, 0, 0, 0.6);
            transform: translateZ(25px);
        }

        .book-cover-img {
            width: 100%;
            height: 100%;
            object-fit: cover;
            transition: transform 0.4s ease;
        }

        .book-card-3d:hover .book-cover-img {
            transform: scale(1.05);
        }

        /* Spine shadow illusion */
        .book-cover-container::after {
            content: '';
            position: absolute;
            left: 0;
            top: 0;
            bottom: 0;
            width: 10px;
            background: linear-gradient(to right, rgba(0,0,0,0.6), transparent);
        }

        .book-info {
            margin-top: 14px;
            transform: translateZ(15px);
        }

        .book-title {
            font-size: 1rem;
            font-weight: 700;
            color: #f8fafc;
            white-space: nowrap;
            overflow: hidden;
            text-overflow: ellipsis;
            margin-bottom: 4px;
        }

        .book-author {
            font-size: 0.85rem;
            color: #cbd5e1;
            font-family: 'Playfair Display', serif;
            font-style: italic;
            margin-bottom: 8px;
        }

        .badge-pill {
            display: inline-flex;
            align-items: center;
            padding: 3px 10px;
            border-radius: 9999px;
            font-size: 0.75rem;
            font-weight: 600;
            background: rgba(168, 85, 247, 0.15);
            border: 1px solid rgba(168, 85, 247, 0.3);
            color: #c084fc;
        }

        .hero-title {
            font-family: 'Playfair Display', serif;
            font-size: 3.2rem;
            font-weight: 700;
            background: linear-gradient(135deg, #ffffff 30%, #c084fc 80%, #6366f1 100%);
            -webkit-background-clip: text;
            -webkit-text-fill-color: transparent;
            text-align: center;
            margin-bottom: 0.2rem;
        }

        .hero-subtitle {
            text-align: center;
            color: #94a3b8;
            font-size: 1.05rem;
            margin-bottom: 2.5rem;
        }
        div[data-testid="stPopover"] {
            position: fixed !important;
            bottom: 40px !important;
            right: 40px !important;
            width: 70px !important;  /* Force the container to be small */
            height: 70px !important; /* Force the container to be small */
            z-index: 99999 !important;
        }
        
        /* 2. Style the Button to fill that small circular container */
        div[data-testid="stPopover"] > button {
            border-radius: 50% !important;
            width: 100% !important;
            height: 100% !important;
            background: linear-gradient(135deg, #a855f7, #6366f1) !important;
            border: none !important;
            box-shadow: 0 10px 25px rgba(168, 85, 247, 0.6) !important;
            color: white !important;
            padding: 0 !important;
            display: flex !important;
            align-items: center !important;
            justify-content: center !important;
            transition: transform 0.3s cubic-bezier(0.175, 0.885, 0.32, 1.275) !important;
        }

        /* 3. Hide the small dropdown arrow Streamlit automatically adds */
        div[data-testid="stPopover"] > button svg {
            display: none !important;
        }

        /* Enlarge emoji inside button */
        div[data-testid="stPopover"] > button p {
            font-size: 32px !important;
            margin: 0 !important;
            padding: 0 !important;
            line-height: 1 !important;
        }

        div[data-testid="stPopover"] > button:hover {
            transform: scale(1.1) translateY(-5px) !important;
        }

        /* 4. Style the Popover Chat Window */
        div[data-testid="stPopoverBody"] {
            width: 400px !important;
            height: 600px !important;
            background: rgba(10, 6, 20, 0.95) !important;
            backdrop-filter: blur(16px) !important;
            border: 1px solid rgba(168, 85, 247, 0.3) !important;
            border-radius: 16px !important;
            padding: 1.5rem !important;
            box-shadow: 0 20px 40px rgba(0,0,0,0.8) !important;
            position: absolute !important;
            bottom: 80px !important; 
            right: 0px !important;
        }
    </style>
    """, unsafe_allow_html=True)

# ==========================================
# 4. DATA PIPELINE & MODEL ENGINE
# ==========================================
@st.cache_data(show_spinner=False)
def load_and_process_data():
    books = pd.read_csv('data/Books.csv', low_memory=False)
    ratings = pd.read_csv('data/Ratings.csv')
    
    # Popularity-based Model
    ratings_with_name = ratings.merge(books, on='ISBN')
    num_rating_df = ratings_with_name.groupby('Book-Title').count()['Book-Rating'].reset_index()
    num_rating_df.rename(columns={'Book-Rating': 'num_ratings'}, inplace=True)

    avg_rating_df = ratings_with_name.groupby('Book-Title')['Book-Rating'].mean().reset_index()
    avg_rating_df.rename(columns={'Book-Rating': 'avg_rating'}, inplace=True)

    popular_df = num_rating_df.merge(avg_rating_df, on='Book-Title')
    popular_df = popular_df[popular_df['num_ratings'] >= 250].sort_values('avg_rating', ascending=False).head(50)
    popular_df = popular_df.merge(books, on='Book-Title').drop_duplicates('Book-Title')[
        ['Book-Title', 'Book-Author', 'Image-URL-M', 'num_ratings', 'avg_rating']
    ]

    # Collaborative Filtering Model
    user_counts = ratings_with_name.groupby('User-ID').count()['Book-Rating'] > 200
    experienced_users = user_counts[user_counts].index
    filtered_ratings = ratings_with_name[ratings_with_name['User-ID'].isin(experienced_users)]

    book_counts = filtered_ratings.groupby('Book-Title').count()['Book-Rating'] >= 50
    famous_books = book_counts[book_counts].index
    final_ratings = filtered_ratings[filtered_ratings['Book-Title'].isin(famous_books)]

    pt = final_ratings.pivot_table(index='Book-Title', columns='User-ID', values='Book-Rating')
    pt.fillna(0, inplace=True)

    similarity_scores = cosine_similarity(pt)
    return books, popular_df, pt, similarity_scores

def recommend_engine(book_name, pt, similarity_scores, books):
    try:
        index = np.where(pt.index == book_name)[0][0]
        similar_items = sorted(
            list(enumerate(similarity_scores[index])),
            key=lambda x: x[1],
            reverse=True
        )[1:5]

        data = []
        for i in similar_items:
            matched_title = pt.index[i[0]]
            temp_df = books[books['Book-Title'] == matched_title].drop_duplicates('Book-Title')
            
            title = temp_df['Book-Title'].values[0]
            author = temp_df['Book-Author'].values[0]
            image = temp_df['Image-URL-M'].values[0]
            score = round(i[1] * 100, 1)

            image = str(image).replace("http://", "https://")
            data.append({"title": title, "author": author, "image": image, "match": score})

        return data
    except Exception:
        return []

@st.cache_resource(show_spinner=False)
def load_rag_pipeline():
    if get_rag_chain is None:
        raise RuntimeError(
            f"RAG import failed: {rag_import_error}"
        )

    return get_rag_chain()

# ==========================================
# 5. UI COMPONENT GENERATOR
# ==========================================
def render_3d_card(title, author, image_url, badge_text):
    safe_image = str(image_url).replace("http://", "https://")
    card_html = f"""
    <div class="book-card-3d-wrap">
        <div class="book-card-3d">
            <div class="book-cover-container">
                <img class="book-cover-img" src="{safe_image}" onerror="this.src='https://images.unsplash.com/photo-1544947950-fa07a98d237f?auto=format&fit=crop&q=80&w=400'" alt="{title}" />
            </div>
            <div class="book-info">
                <div class="book-title" title="{title}">{title}</div>
                <div class="book-author">{author}</div>
                <div class="badge-pill">{badge_text}</div>
            </div>
        </div>
    </div>
    """
    st.markdown(card_html, unsafe_allow_html=True)

# ==========================================
# 6. MAIN APPLICATION EXECUTION
# ==========================================
render_3d_background()
inject_custom_styles()

st.markdown('<div class="hero-title">Lumina Cognition</div>', unsafe_allow_html=True)
st.markdown('<div class="hero-subtitle">Next-Generation Collaborative Vector Book Discovery</div>', unsafe_allow_html=True)

with st.spinner("✨ Initializing vector spaces and indexing catalog..."):
    try:
        books, popular_df, pt, similarity_scores = load_and_process_data()
    except FileNotFoundError:
        st.error("Missing dataset files! Ensure `Books.csv` and `Ratings.csv` are in the app directory.")
        st.stop()

# Initialize 3 tabs properly
tab_explore, tab_popular = st.tabs([
    "🔮 Vector Recommendation", 
    "🏆 Hall of Fame (Top 50)"
])

# ----------------- TAB 1: RECOMMENDATIONS -----------------
with tab_explore:
    col_input, col_action = st.columns([4, 1])
    with col_input:
        selected_book = st.selectbox(
            "Select or search for a literary baseline:",
            options=pt.index.values,
            help="Choose a book from the collaborative space to compute nearest cosine vectors."
        )
    with col_action:
        st.markdown("<div style='height: 28px;'></div>", unsafe_allow_html=True)
        generate_btn = st.button("Synthesize Match ✨", use_container_width=True)

    if generate_btn or selected_book:
        st.markdown(f"#### 🛰️ Resonant Matches for: *{selected_book}*")
        recommendations = recommend_engine(selected_book, pt, similarity_scores, books)

        if recommendations:
            cols = st.columns(4)
            for idx, rec in enumerate(recommendations):
                with cols[idx]:
                    render_3d_card(
                        title=rec["title"],
                        author=rec["author"],
                        image_url=rec["image"],
                        badge_text=f"Vector Match: {rec['match']}%"
                    )
        else:
            st.warning("No latent neighbors located for this title.")

# ----------------- TAB 2: POPULARITY SHOWCASE -----------------
with tab_popular:
    st.markdown("#### 🌟 Top 50 Globally Celebrated Titles (Minimum 250 Ratings)")
    
    batch_size = 5
    for i in range(0, len(popular_df), batch_size):
        cols = st.columns(batch_size)
        batch = popular_df.iloc[i : i + batch_size]
        for col, (_, row) in zip(cols, batch.iterrows()):
            with col:
                render_3d_card(
                    title=row["Book-Title"],
                    author=row["Book-Author"],
                    image_url=row["Image-URL-M"],
                    badge_text=f"★ {row['avg_rating']:.2f} ({int(row['num_ratings'])})"
                )

# ----------------- TAB 3: RAG CHATBOT -----------------
# with tab_chat:
#     st.markdown("### 💬 Chat with Lumina AI Concierge")
#     st.caption("Ask questions like: *'Find me a dystopian book similar to 1984'* or *'Suggest historical novels with high drama.'*")

#     rag_chain, retriever = load_rag_pipeline()

#     if rag_chain is None or retriever is None:
#         st.info("⚠️ RAG Engine offline. Ensure `rag_engine.py` and the vector index `./chroma_books_db` exist.")
#     else:
#         if "rag_messages" not in st.session_state:
#             st.session_state.rag_messages = [
#                 {"role": "assistant", "content": "Greetings! I'm Lumina. Tell me what mood, theme, or author you're looking for, and I'll find matches from our index."}
#             ]

#         for message in st.session_state.rag_messages:
#             with st.chat_message(message["role"]):
#                 st.markdown(message["content"])

#         if user_prompt := st.chat_input("Ask about themes, characters, or get personalized recommendations..."):
#             st.session_state.rag_messages.append({"role": "user", "content": user_prompt})
#             with st.chat_message("user"):
#                 st.markdown(user_prompt)

#             retrieved_docs = retriever.invoke(user_prompt)

#             with st.chat_message("assistant"):
#                 response_container = st.empty()
#                 with st.spinner("Scanning literary archives..."):
#                     response_text = rag_chain.invoke(user_prompt)
#                     response_container.markdown(response_text)

#                 # Render retrieved book references with safety checks
#                 if retrieved_docs:
#                     st.markdown("##### 📖 Referenced Titles from Index")
#                     cols = st.columns(min(len(retrieved_docs), 4))
#                     for idx, doc in enumerate(retrieved_docs[:4]):
#                         meta = doc.metadata
#                         with cols[idx]:
#                             render_3d_card(
#                                 title=meta.get('title', 'Unknown'),
#                                 author=meta.get('author', 'Unknown'),
#                                 image_url=meta.get('image', ''),
#                                 badge_text=f"Genre: {str(meta.get('categories', 'General'))[:18]}"
#                             )

#             st.session_state.rag_messages.append({"role": "assistant", "content": response_text})
            
            
# ==========================================
# 7. FLOATING AI CONCIERGE WIDGET
# ==========================================
rag_chain, retriever = load_rag_pipeline()

# Create the popover button (styled as a floating circle via CSS)
with st.popover("💬"):
    st.markdown("### 💬 Biblo AI")
    st.caption("Ask me about themes, characters, or recommendations.")
    st.divider()

    if rag_chain is None or retriever is None:
        st.info("⚠️ RAG Engine offline. Check vector index and API keys.")
    else:
        # Create a scrollable container for the chat history
        chat_container = st.container(height=400, border=False)
        
        if "rag_messages" not in st.session_state:
            st.session_state.rag_messages = [
                {"role": "assistant", "content": "Greetings! Tell me what mood or theme you're looking for."}
            ]

        with chat_container:
            for message in st.session_state.rag_messages:
                with st.chat_message(message["role"]):
                    st.markdown(message["content"])

        # Chat Input (Supported inside popovers in Streamlit 1.37+)
        if user_prompt := st.chat_input("Ask about a book..."):
            st.session_state.rag_messages.append({"role": "user", "content": user_prompt})
            
            with chat_container:
                with st.chat_message("user"):
                    st.markdown(user_prompt)

                with st.chat_message("assistant"):
                    response_container = st.empty()
                    with st.spinner("Scanning archives..."):
                        response_text = rag_chain.invoke(user_prompt)
                        response_container.markdown(response_text)
                        
            st.session_state.rag_messages.append({"role": "assistant", "content": response_text})
            st.rerun()
