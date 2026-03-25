import streamlit as st
import pandas as pd
import numpy as np
import re
import nltk
from nltk.corpus import stopwords

nltk.download('stopwords', quiet=True)
STOP = set(stopwords.words('english'))

st.set_page_config(page_title="Insurer Analysis", page_icon="🏢", layout="wide")

@st.cache_data
def load_data():
    df = pd.read_csv('insurance_reviews_processed.csv')
    df['date_publication'] = pd.to_datetime(df['date_publication'], dayfirst=True, errors='coerce')
    return df

def clean(text):
    if not isinstance(text, str): return ''
    text = text.lower()
    text = re.sub(r'[^a-z\s]', ' ', text)
    return ' '.join(t for t in text.split() if t not in STOP and len(t) > 2)

@st.cache_resource
def load_summarizer():
    from transformers import pipeline
    return pipeline('summarization', model='facebook/bart-large-cnn',
                    max_length=120, min_length=40, truncation=True)

@st.cache_resource
def load_sbert():
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer('all-MiniLM-L6-v2')

@st.cache_data
def get_embeddings(texts):
    model = load_sbert()
    return model.encode(texts, batch_size=128, show_progress_bar=False)

df = load_data()

st.title("🏢 Insurer Analysis Dashboard")
st.write("Explore performance metrics, search reviews, and get summaries by insurer.")

st.sidebar.header("Filters")
all_insurers = sorted(df['assureur'].dropna().unique())
selected_ins = st.sidebar.multiselect("Insurer(s)", all_insurers, default=all_insurers[:5])

all_products = sorted(df['produit'].dropna().unique())
selected_prod = st.sidebar.multiselect("Product(s)", all_products, default=all_products)

star_range = st.sidebar.slider("Star range", 1, 5, (1, 5))

df_filt = df[
    df['assureur'].isin(selected_ins) &
    df['produit'].isin(selected_prod) &
    df['note'].between(star_range[0], star_range[1])
].copy()

st.caption(f"{len(df_filt)} reviews matching filters")

tab1, tab2, tab3, tab4 = st.tabs([" Metrics", " Search", " Summary", " Q&A (RAG)"])
with tab1:
    st.subheader("Average rating by insurer")
    if df_filt.empty:
        st.info("No data for current filters.")
    else:
        avg_ins = (df_filt.groupby('assureur')['note']
                   .agg(['mean', 'count'])
                   .rename(columns={'mean': 'avg_rating', 'count': 'nb_reviews'})
                   .sort_values('avg_rating'))
        st.bar_chart(avg_ins['avg_rating'])

        st.subheader("Average rating by product")
        avg_prod = df_filt.groupby('produit')['note'].mean().sort_values()
        st.bar_chart(avg_prod)

        st.subheader("Star distribution")
        star_dist = df_filt['note'].value_counts().sort_index()
        st.bar_chart(star_dist)

        st.subheader("Detailed table")
        st.dataframe(avg_ins.reset_index().sort_values('avg_rating', ascending=False),
                     use_container_width=True)

        st.subheader("Average rating : insurer × product")
        pivot = df_filt.pivot_table(values='note', index='assureur',
                                    columns='produit', aggfunc='mean').round(2)
        st.dataframe(pivot.style.background_gradient(cmap='RdYlGn', vmin=1, vmax=5),
                     use_container_width=True)
with tab2:
    st.subheader("Review Search")

    search_mode = st.radio("Search mode", ["Keyword search", "Semantic search"], horizontal=True)
    query = st.text_input("Enter your search query")
    top_k = st.slider("Number of results", 5, 30, 10)

    if query:
        if search_mode == "Keyword search":
            mask = df_filt['avis_en'].str.contains(query, case=False, na=False)
            results = df_filt[mask].head(top_k)[['assureur', 'produit', 'note', 'avis_en', 'date_publication']]
            st.write(f"{mask.sum()} results found")
            st.dataframe(results, use_container_width=True)

        else:
            with st.spinner("Computing embeddings..."):
                from sklearn.metrics.pairwise import cosine_similarity
                sbert = load_sbert()
                q_emb = sbert.encode([query])
                corpus_texts = df_filt['avis_en'].fillna('').tolist()
                corpus_emb   = get_embeddings(tuple(corpus_texts))
                sims = cosine_similarity(q_emb, corpus_emb)[0]
                top_idx = np.argsort(sims)[::-1][:top_k]
                results = df_filt.iloc[top_idx][['assureur', 'produit', 'note', 'avis_en']].copy()
                results['similarity'] = sims[top_idx].round(3)
                st.dataframe(results, use_container_width=True)

with tab3:
    st.subheader("Summary by Insurer")

    ins_choice = st.selectbox("Choose an insurer", selected_ins if selected_ins else all_insurers)
    max_reviews_sum = st.slider("Reviews to summarize", 10, 100, 30)

    if st.button("Generate summary", type="primary"):
        subset = df[df['assureur'] == ins_choice]['avis_en'].dropna().head(max_reviews_sum).tolist()
        if not subset:
            st.warning("No reviews for this insurer.")
        else:
            combined = ' '.join(subset)[:3000]
            with st.spinner("Summarizing..."):
                summarizer = load_summarizer()
                summary = summarizer(combined)[0]['summary_text']

            st.success("Summary generated !")
            st.write(f"**{ins_choice}** - based on {len(subset)} reviews")
            st.info(summary)

            ins_df = df[df['assureur'] == ins_choice]
            c1, c2, c3 = st.columns(3)
            c1.metric("Average rating",  f"{ins_df['note'].mean():.2f} / 5")
            c2.metric("Total reviews",   len(ins_df))
            c3.metric("% positive (4-5★)", f"{(ins_df['note'] >= 4).mean()*100:.0f}%")
with tab4:
    st.subheader("Q&A on Reviews (RAG)")
    st.caption("Ask a question - the system finds relevant reviews and generates an answer.")

    question = st.text_input("Your question", placeholder="What do customers say about pricing?")
    n_context = st.slider("Number of reviews to use as context", 3, 15, 5)

    if st.button("Ask", type="primary") and question:
        with st.spinner("Searching relevant reviews..."):
            from sklearn.metrics.pairwise import cosine_similarity
            sbert     = load_sbert()
            q_emb     = sbert.encode([question])
            pool      = df_filt['avis_en'].fillna('').tolist()
            pool_emb  = get_embeddings(tuple(pool))
            sims      = cosine_similarity(q_emb, pool_emb)[0]
            top_idx   = np.argsort(sims)[::-1][:n_context]
            context_reviews = [pool[i] for i in top_idx]
            context_str     = '\n\n'.join([f'Review {i+1}: {r[:300]}' for i, r in enumerate(context_reviews)])

        with st.spinner("Generating answer..."):
            from transformers import pipeline
            qa_pipe = pipeline('text2text-generation', model='google/flan-t5-base', max_new_tokens=200)
            prompt  = f"Based on these insurance reviews, answer the question.\n\nReviews:\n{context_str}\n\nQuestion: {question}\n\nAnswer:"
            answer  = qa_pipe(prompt)[0]['generated_text']

        st.subheader("Answer")
        st.success(answer)

        st.subheader("Context reviews used")
        ctx_df = df_filt.iloc[top_idx][['assureur', 'note', 'avis_en']].copy()
        ctx_df['similarity'] = sims[top_idx].round(3)
        st.dataframe(ctx_df, use_container_width=True)
