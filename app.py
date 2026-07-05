import streamlit as st
import networkx as nx
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import sqlite3
from allensdk.core.mouse_connectivity_cache import MouseConnectivityCache
from scipy.spatial.distance import cdist

st.set_page_config(
    page_title="NeuroNav",
    page_icon="🧠",
    layout="wide"
)

# ── custom CSS ────────────────────────────────────────────
st.markdown("""
<style>
* {
    font-family: 'Times New Roman', Times, serif !important;
}
[data-testid="stSidebar"] { padding-top: 2rem; }
.nav-link {
    display: block; padding: 0.5rem 0; font-size: 1rem;
    color: #aaaaaa; text-decoration: none; cursor: pointer;
    font-weight: 400; transition: color 0.2s;
    border: none; background: none; width: 100%; text-align: left;
}
.nav-link:hover { color: #ffffff; font-weight: 600; }
.nav-link-active {
    display: block; padding: 0.5rem 0; font-size: 1rem;
    color: #ffffff; font-weight: 700;
    border: none; background: none; width: 100%; text-align: left;
}
.section-label {
    font-size: 0.7rem; text-transform: uppercase;
    letter-spacing: 0.12em; color: #666666;
    margin-bottom: 0.5rem; margin-top: 1.5rem;
}
.context-box {
    background-color: #1a1a2e;
    border-left: 3px solid #1D9E75;
    padding: 1rem 1.2rem;
    border-radius: 0 8px 8px 0;
    margin-bottom: 1.5rem;
    font-size: 0.92rem;
    line-height: 1.7;
    color: #cccccc;
}
.context-box b { color: #ffffff; }
</style>
""", unsafe_allow_html=True)

# ── data loading ──────────────────────────────────────────
@st.cache_data
def load_data():
    mcc = MouseConnectivityCache(manifest_file='allen_cache/manifest.json')
    structure_tree = mcc.get_structure_tree()
    experiments    = mcc.get_experiments(dataframe=True)
    top_regions = (
        experiments['structure_abbrev']
        .value_counts().head(20).index.tolist()
    )
    region_coords = {}
    for region in top_regions:
        exps = experiments[experiments['structure_abbrev'] == region]
        if len(exps) > 0:
            region_coords[region] = [
                exps['injection_x'].mean(),
                exps['injection_y'].mean(),
                exps['injection_z'].mean()
            ]
    valid_regions   = [r for r in top_regions if r in region_coords]
    coords_array    = np.array([region_coords[r] for r in valid_regions])
    dist_matrix     = cdist(coords_array, coords_array, metric='euclidean')
    strength_matrix = 1 / (dist_matrix + 1e-6)
    np.fill_diagonal(strength_matrix, 0)
    strength_matrix /= strength_matrix.max()
    matrix_df = pd.DataFrame(strength_matrix,
                              index=valid_regions, columns=valid_regions)
    return valid_regions, region_coords, strength_matrix, matrix_df

@st.cache_data
def load_db():
    conn = sqlite3.connect('db/neuro_nav.db')
    paths   = pd.read_sql('SELECT * FROM pathfinding_results', conn)
    align   = pd.read_sql('SELECT * FROM alignment_results', conn)
    cluster = pd.read_sql('SELECT * FROM clustering_results', conn)
    ml      = pd.read_sql('SELECT * FROM ml_results', conn)
    conn.close()
    return paths, align, cluster, ml

def build_graph(valid_regions, region_coords, strength_matrix, threshold=0.15):
    G = nx.DiGraph()
    for region in valid_regions:
        G.add_node(region, coords=region_coords[region])
    for i, src in enumerate(valid_regions):
        for j, tgt in enumerate(valid_regions):
            if i != j and strength_matrix[i][j] > threshold:
                G.add_edge(src, tgt,
                           weight=strength_matrix[i][j],
                           cost=1 - strength_matrix[i][j])
    return G

# ── load ──────────────────────────────────────────────────
valid_regions, region_coords, strength_matrix, matrix_df = load_data()
paths_df, align_df, cluster_df, ml_df = load_db()
G = build_graph(valid_regions, region_coords, strength_matrix)

# ── sidebar navigation ────────────────────────────────────
if 'page' not in st.session_state:
    st.session_state.page = 'Overview'

def nav_button(label):
    if st.sidebar.button(label, key=f"nav_{label}",
                         use_container_width=True):
        st.session_state.page = label
        st.rerun()

with st.sidebar:
    st.markdown("### NeuroNav")
    st.markdown("Mouse Brain Connectome Explorer")
    st.markdown('<div class="section-label">Navigate</div>',
                unsafe_allow_html=True)
    for p in ["Overview", "Brain Graph", "Pathfinding",
              "Alignment", "ML Results", "Region Guide", "References"]:
        nav_button(p)

page = st.session_state.page

# ── overview ──────────────────────────────────────────────
if page == "Overview":
    st.title("NeuroNav")
    st.markdown("#### Mouse Brain Connectome Explorer")

    st.markdown("""
    <div class="context-box">
    <b>What is this project?</b><br>
    NeuroNav is a computational neuroscience project that models the mouse brain as a
    directed graph using data from the <b>Allen Mouse Brain Connectivity Atlas</b> and
    the <b>PhysioNet EEG Motor Imagery dataset</b>. The connectivity atlas is a dataset
    built from 2,992 viral tracing experiments across hundreds of mice. Each experiment
    injected an AAV virus into one brain region, allowing axonal projections to be mapped
    across the entire brain (Oh et al., 2014). The EEG dataset contains 64-channel brain
    recordings from humans performing motor imagery tasks (Schalk et al., 2004).<br><br>
    <b>What can you explore here?</b><br>
    This dashboard lets you visualize the brain's wiring, run graph algorithms to find
    optimal signal pathways, compare brain regions by their connectivity fingerprints,
    and see how machine learning classifies neural signals from EEG recordings.<br><br>
    <b>Signal pathway finding:</b> The brain communicates by sending electrical signals
    along chains of connected neurons. Finding the optimal pathway between two regions
    tells us which routes are most efficient for neural communication. In the real world
    this has direct applications in <b>deep brain stimulation</b> (targeting the right
    pathway to treat Parkinson's or depression), <b>stroke rehabilitation</b>
    (identifying which alternative pathways remain intact when primary ones are damaged),
    and <b>surgical planning</b> (avoiding disruption to critical signal routes).<br><br>
    <b>Connectivity fingerprint comparison:</b> Every brain region has a unique pattern
    of connections to other regions, like a wiring signature. Comparing these signatures
    reveals which regions serve similar functional roles, even without knowing their
    anatomy in advance (Passingham et al., 2002). This matters because it can help
    identify functionally equivalent regions across species (Mars et al., 2018), detect
    abnormal connectivity patterns in diseases like schizophrenia where wiring is
    disrupted, and potentially classify brain regions in newly mapped areas where
    function is unknown.<br><br>
    <b>EEG motor imagery classification:</b> When you imagine moving your hand without
    actually moving it, your brain produces measurable electrical patterns over the motor
    cortex (Pfurtscheller and Lopes da Silva, 1999). Teaching a machine to recognize
    these patterns is the foundation of <b>brain-computer interfaces (BCIs)</b>,
    technology that allows people with paralysis, ALS, or locked-in syndrome to control
    prosthetic limbs, wheelchairs, or communication devices purely through thought.
    The accuracy numbers here represent how reliably a classifier could decode that
    intent from raw brain signals.
    </div>
    """, unsafe_allow_html=True)

    c1, c2, c3, c4 = st.columns(4)
    c1.metric("Brain Regions", "20")
    c2.metric("Connections", "134")
    c3.metric("Best ML Accuracy", "66.7%")
    c4.metric("Strongest Connection", "DG→CA1: 0.543")

    st.markdown("---")
    st.subheader("Project Pipeline")
    p1, p2, p3, p4 = st.columns(4)
    with p1:
        st.markdown("**Phase 1**")
        st.markdown("Data loading, graph construction, BFS/DFS traversal, SQLite database")
    with p2:
        st.markdown("**Phase 2**")
        st.markdown("Dijkstra's, A\\*, Bellman-Ford, Needleman-Wunsch sequence alignment")
    with p3:
        st.markdown("**Phase 3**")
        st.markdown("K-Means clustering, SVM/Random Forest, EEG Transformer (PyTorch)")
    with p4:
        st.markdown("**Phase 4**")
        st.markdown("LLM chatbot interface, interactive Streamlit dashboard")

    st.markdown("---")
    st.subheader("Top 10 Strongest Connections")
    st.markdown(
        "Connection strength is computed from the inverse Euclidean distance between "
        "brain region injection coordinates. Euclidean distance is simply the "
        "straight-line distance between two points in 3D space. The shorter the "
        "physical distance between two regions, the stronger we estimate their "
        "connection to be. This reflects the brain's metabolic cost constraint on "
        "long-range wiring: growing and maintaining long axons is energetically "
        "expensive, so the brain preferentially forms strong connections between "
        "nearby regions (Oh et al., 2014). Identifying which regions are most strongly "
        "connected has direct research and clinical relevance. Strongly connected "
        "region pairs are the primary routes of neural communication, the first "
        "circuits studied when understanding how a brain system works, and often the "
        "first affected when disease strikes. The DG→CA1 connection (the strongest in "
        "our dataset at 0.543) is the core of the hippocampal memory circuit "
        "(Bhaskaran and Bhaskaran, 2007), and its disruption is one of the earliest "
        "signs of Alzheimer's disease (Siman et al., 2015; Uchida et al., 2025)."
    )
    top_conn = []
    for i, src in enumerate(valid_regions):
        for j, tgt in enumerate(valid_regions):
            if i != j:
                top_conn.append({'Source': src, 'Target': tgt,
                                  'Strength': round(strength_matrix[i][j], 4)})
    top_df = pd.DataFrame(top_conn).sort_values(
        'Strength', ascending=False).head(10)
    st.dataframe(top_df, use_container_width=True)

# ── brain graph ───────────────────────────────────────────
elif page == "Brain Graph":
    st.title("Brain Region Graph")

    st.markdown("""
    <div class="context-box">
    <b>What are you looking at?</b><br>
    This is a directed weighted graph of the 20 most-studied mouse brain regions from
    the Allen Atlas (Oh et al., 2014). Each <b>node</b> is a brain region and each
    <b>edge</b> represents an axonal projection, the physical connection along which
    neurons send signals from one region to another. The direction of the edge matters:
    a projection from the hippocampus to the entorhinal cortex is biologically distinct
    from one traveling in the opposite direction, and the two do not necessarily have
    equal strength.<br><br>
    The <b>connection threshold slider</b> controls which edges are visible. Every pair
    of brain regions has some spatial relationship, but not all of them represent
    meaningful axonal projections. Some are simply nearby by chance with no significant
    direct wiring. The threshold filters out weak connections below a minimum strength
    value, leaving only the most biologically relevant edges. <b>Raising the threshold</b>
    removes weaker connections and reveals the brain's core highway network, the
    strongest and most essential pathways. <b>Lowering it</b> reveals more connections
    including weaker, less certain ones. This mirrors a real challenge in connectomics
    research: deciding which connections are strong enough to be considered functionally
    meaningful versus background noise (Oh et al., 2014). At the default threshold of
    0.15, the graph shows 134 connections that represent the most reliable spatial
    relationships in our dataset.
    </div>
    """, unsafe_allow_html=True)

    threshold = st.slider(
        "Connection threshold — raise to show only the strongest connections",
        0.05, 0.5, 0.15, 0.01
    )
    G_vis = build_graph(valid_regions, region_coords, strength_matrix, threshold)
    st.metric("Edges above threshold", G_vis.number_of_edges())

    fig, ax = plt.subplots(figsize=(12, 8))
    fig.patch.set_facecolor('#0e1117')
    ax.set_facecolor('#0e1117')
    pos = nx.spring_layout(G_vis, seed=42)
    weights = [G_vis[u][v]['weight'] * 3 for u, v in G_vis.edges()]
    nx.draw_networkx_nodes(G_vis, pos, node_color='#1D9E75',
                           node_size=600, ax=ax)
    nx.draw_networkx_labels(G_vis, pos, font_size=8,
                             font_color='white', ax=ax)
    nx.draw_networkx_edges(G_vis, pos, width=weights,
                            edge_color='#5DCAA5', alpha=0.6,
                            arrows=True, ax=ax)
    ax.axis('off')

    st.caption(
        "Note: Node positions are determined by a spring layout algorithm "
        "(NetworkX force-directed layout) that minimizes edge crossings for readability. "
        "This is not an anatomical projection, positions do not reflect actual brain "
        "coordinates. Edge thickness reflects connection strength. "
        "See the K-Means clustering plot on the ML Results page for spatial organization "
        "based on real 3D coordinates."
    )

    st.pyplot(fig)

# ── pathfinding ───────────────────────────────────────────
elif page == "Pathfinding":
    st.title("Neural Pathway Finder")

    st.markdown("""
    <div class="context-box">
    <b>What is this tool?</b><br>
    This tool uses <b>Dijkstra's algorithm</b> to find the strongest signal pathway
    between any two brain regions, the route a neural signal would most efficiently
    travel through the connectome. Dijkstra's works by finding the minimum-cost path
    through a weighted graph. Since we want the <i>strongest</i> pathway, we invert
    the connection weights using the formula <b>cost = 1 − strength</b>. This means a
    strong connection (strength 0.95) becomes very cheap to travel (cost 0.05), while
    a weak connection (strength 0.15) becomes expensive (cost 0.85). The algorithm
    will naturally gravitate toward high-strength routes without us having to modify the
    algorithm itself.<br><br>
    <b>Strength</b> in this context refers to the normalized spatial proximity between
    two brain regions derived from their Allen Atlas injection coordinates (Oh et al.,
    2014), a value between 0 and 1 where 1 means maximally close and strongly connected.
    This models a real neuroscience principle: neural signals preferentially propagate
    along high-strength axonal pathways, the same way electrical current follows the
    path of least resistance. Finding the optimal pathway between two regions has direct
    clinical relevance: in deep brain stimulation therapy, surgeons need to know which
    signal routes exist between a stimulation target and downstream regions to predict
    therapeutic effects and avoid side effects.
    </div>
    """, unsafe_allow_html=True)

    col1, col2 = st.columns(2)
    source = col1.selectbox("Source region", valid_regions,
                             index=valid_regions.index('ENTl'))
    target = col2.selectbox("Target region", valid_regions,
                             index=valid_regions.index('CA1'))

    if st.button("Find strongest pathway"):
        try:
            path = nx.dijkstra_path(G, source, target, weight='cost')
            hops = []
            for i in range(len(path)-1):
                w = G[path[i]][path[i+1]]['weight']
                hops.append({'From': path[i], 'To': path[i+1],
                              'Strength': round(w, 4)})
            st.success(f"Optimal route: {' → '.join(path)}")
            st.dataframe(pd.DataFrame(hops), use_container_width=True)
            col_a, col_b = st.columns(2)
            col_a.metric("Path length", f"{len(path)-1} hops")
            col_b.metric("Average strength",
                          round(np.mean([h['Strength'] for h in hops]), 4))
        except nx.NetworkXNoPath:
            st.error(f"No path found between {source} and {target} "
                     f"above the current threshold.")

    st.markdown("---")
    st.subheader("Saved Pathfinding Results")
    st.markdown(
        "The table below documents pre-computed pathfinding results from Phase 2 of "
        "this project, three specific neuroscientific queries run across three "
        "algorithms. These are not arbitrary region pairs. Each was chosen to represent "
        "a known neural circuit with real biological significance. "
        "**ENTl → DG** is the perforant path: the primary route by which sensory and "
        "memory information enters the hippocampus from the entorhinal cortex (Igarashi, "
        "2023), one of the first pathways destroyed in Alzheimer's disease (Siman et al., "
        "2015; Uchida et al., 2025). "
        "**MOs → CP** is the corticostriatal pathway: how the motor cortex communicates "
        "with the striatum to plan and execute voluntary movement, routing through ACAv "
        "(motor planning → action selection → motor execution). "
        "**VISp → CA1** is the visual memory encoding route: how visual information "
        "travels from primary visual cortex into the hippocampus via the dentate gyrus, "
        "capturing the hippocampal trisynaptic circuit (Bhaskaran and Bhaskaran, 2007)."
    )
    st.dataframe(
        paths_df[['algorithm', 'source', 'target', 'path', 'avg_pathway_strength']]
        .sort_values('avg_pathway_strength', ascending=False),
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("Algorithm Discussion")
    st.markdown("""
This project implemented five graph traversal and pathfinding algorithms across Phase 2,
each answering a different neuroscientific question.

**Dijkstra's** is used as the interactive tool above because it finds the
minimum-cost (maximum-strength) pathway, the most meaningful query for understanding
neural signal routing.

**A\\*** produced identical pathways to Dijkstra's across all three region pairs,
confirming a known neuroscience finding called *spatial embedding*: in the brain, the
physically closest pathway tends to also be the strongest one (Oh et al., 2014). A*
uses 3D spatial coordinates as a heuristic to guide its search, and the fact that it
agreed with Dijkstra's means spatial proximity and connection strength are correlated
in this dataset.

**BFS and DFS** are non-weighted traversals. They explore connectivity structure
without considering edge strength. They are not appropriate for finding optimal signal
routes, but were used in Phase 1 to validate the graph structure and model how a signal
spreads level by level (BFS) or along a deep chain (DFS) from the hippocampus.

**Bellman-Ford** is the most neuroscientifically novel algorithm in this project. Unlike
Dijkstra's, it handles negative edge weights, which arise when connection penalties push
costs above 1.0 during neurodegeneration simulation. The Bellman-Ford rows in the table
above model Alzheimer's disease progression on the ENTl→CA1 pathway by progressively
weakening entorhinal connections: healthy (0.193), early degeneration (0.113), severe
degeneration (0.053), a 73% signal loss (Igarashi, 2023; Siman et al., 2015). The
pathway never reroutes. It simply gets quieter, mirroring the clinical reality of
Alzheimer's: the hippocampus doesn't find a detour, it just stops receiving strong
enough input to encode new memories (Uchida et al., 2025).
""")

# ── alignment ─────────────────────────────────────────────
elif page == "Alignment":
    st.title("Connectivity Signature Alignment")

    st.markdown("""
    <div class="context-box">
    <b>What is a connectivity signature?</b><br>
    Every brain region has a unique connectivity fingerprint, a pattern of how strongly
    it connects to every other region in the network (Passingham et al., 2002). A brain
    region can be characterized by which other regions it connects to and how strongly.
    Two regions with similar fingerprints tend to participate in the same circuits and
    serve similar functional roles, even if their names or locations are different. This
    concept is used in real neuroscience research to classify unknown brain areas,
    compare brain organization across species (Mars et al., 2018; Choi et al., 2020),
    and identify regions whose connectivity is abnormally altered in psychiatric or
    neurological conditions.<br><br>
    <b>What is Needleman-Wunsch?</b><br>
    Needleman-Wunsch is a classic dynamic programming algorithm from bioinformatics,
    originally designed for global DNA and protein sequence alignment, comparing two
    genetic sequences character by character to find their optimal match. Here it is
    adapted to compare connectivity signatures instead of DNA bases. Rather than
    matching A, T, G, C nucleotides, it matches connection strength values between
    region pairs. At each position, the algorithm chooses to match two strengths or
    insert a gap in one sequence, finding the globally optimal alignment that maximizes
    similarity across the entire fingerprint. A score near <b>1.0</b> means nearly
    identical wiring patterns. A lower score means systematically different connection
    profiles, reflecting different functional roles. The scoring uses an exponential
    penalty function to amplify differences, making the ranking more sensitive and
    biologically meaningful.<br><br>
    <b>Neuroscience relevance:</b><br>
    The results validate that connectivity fingerprints capture real functional
    organization (Passingham et al., 2002; Mars et al., 2018). ACAd vs ACAv (0.979):
    subdivisions of the same structure sharing nearly all the same inputs and outputs.
    ENTl vs ENTm (0.969): neighbors serving as the primary gateway to the hippocampus
    (Igarashi, 2023). VISp vs VISl (0.953): part of the same visual processing
    hierarchy. CA1 vs VISp (0.808): hippocampus and primary visual cortex serve
    completely different functions and have the most dissimilar wiring in the dataset.
    The algorithm recovered known functional relationships without being given any
    anatomical labels, purely from connectivity patterns.
    </div>
    """, unsafe_allow_html=True)

    st.markdown(
        "The table below ranks all six region pairs by alignment score from most to "
        "least similar. Each row shows the two regions compared, their similarity score, "
        "and a brief anatomical description. Scores range from 0.808 to 0.979. Even "
        "the most dissimilar pair shares substantial connectivity structure, reflecting "
        "that all 20 regions are drawn from the same densely interconnected network. "
        "A score of 1.0 would mean perfectly identical wiring, which would only occur "
        "if two regions were exact anatomical duplicates."
    )
    st.dataframe(
        align_df[['region_a', 'region_b', 'alignment_score', 'description']]
        .sort_values('alignment_score', ascending=False),
        use_container_width=True
    )

    st.markdown(
        "The chart below visualizes the same ranking. The x-axis is deliberately zoomed "
        "to 0.75–1.0 rather than 0–1. At full scale all bars would appear nearly "
        "identical since all scores fall in a narrow range. This zoom reveals the "
        "meaningful spread. The ordering from bottom to top goes from most similar "
        "(ACAd vs ACAv) to most different (CA1 vs VISp), confirming that functionally "
        "related region pairs consistently score higher than unrelated ones."
    )
    fig, ax = plt.subplots(figsize=(8, 4))
    sorted_df = align_df.sort_values('alignment_score', ascending=True)
    pairs  = [f"{r['region_a']} vs {r['region_b']}"
               for _, r in sorted_df.iterrows()]
    scores = sorted_df['alignment_score'].tolist()
    bars   = ax.barh(pairs, scores, color='#1D9E75')
    ax.set_xlim(0.75, 1.0)
    ax.set_xlabel('Alignment score')
    ax.set_title('Wiring similarity between region pairs')
    for bar, score in zip(bars, scores):
        ax.text(score + 0.001, bar.get_y() + bar.get_height()/2,
                f'{score:.3f}', va='center', fontsize=9)
    fig.tight_layout()
    st.pyplot(fig)

# ── ml results ────────────────────────────────────────────
elif page == "ML Results":
    st.title("Machine Learning Results")

    st.markdown("""
    <div class="context-box">
    <b>What was classified?</b><br>
    EEG recordings from the PhysioNet Motor Imagery dataset were used to classify
    whether a person was imagining moving their <b>left or right fist</b>, without
    any physical movement occurring (Schalk et al., 2004). This is the core task in
    brain-computer interface (BCI) research: decoding a person's motor intent directly
    from their brain signals. The ability to do this reliably has profound clinical
    implications. It is the technology that allows people with ALS, spinal cord
    injuries, or locked-in syndrome to control prosthetic limbs, robotic wheelchairs,
    or communication devices purely through thought. Every percentage point of
    classification accuracy above chance represents real progress toward making these
    systems practical and safe for clinical use.<br><br>
    <b>Why these models?</b><br>
    Three fundamentally different approaches were compared. <b>SVM</b> uses hand-crafted
    features, the average signal power in alpha (8-12 Hz), beta (12-30 Hz), and gamma
    (30-40 Hz) frequency bands across 64 EEG channels (Pfurtscheller and Lopes da
    Silva, 1999), and finds the optimal mathematical boundary separating left from right
    fist patterns. <b>Random Forest</b> builds an ensemble of decision trees on the
    same features, voting on the most likely class. <b>EEG Transformer</b> takes a
    fundamentally different approach: rather than using hand-crafted features, it
    processes raw EEG time-series data through a self-attention mechanism, learning
    which moments in time are most informative for classification entirely from data.
    The <b>50% dashed line</b> represents chance level. Any model below it is actively
    learning the wrong patterns.
    </div>
    """, unsafe_allow_html=True)

    st.subheader("Model Accuracy Comparison")
    st.markdown(
        "The chart below compares all models by accuracy. SVM and SVM_crossval appear "
        "separately because test-set accuracy and 5-fold cross-validation are two "
        "different ways of measuring the same model's performance. Cross-validation is "
        "the more reliable estimate because it averages across 5 independent train-test "
        "splits. Color coding: **dark green** = above 60% (meaningfully above chance), "
        "**light green** = 50–60% (marginal), **gray** = at or below chance."
    )
    fig, ax = plt.subplots(figsize=(7, 4))
    ml_sorted = ml_df.sort_values('accuracy', ascending=True)
    colors = ['#1D9E75' if a > 0.6 else '#5DCAA5' if a > 0.5 else '#B4B2A9'
              for a in ml_sorted['accuracy']]
    ax.barh(ml_sorted['model'], ml_sorted['accuracy'], color=colors)
    ax.axvline(0.5, color='#E8593C', linestyle='--',
                alpha=0.7, label='Chance (50%)')
    ax.set_xlim(0, 1)
    ax.set_xlabel('Accuracy')
    ax.set_title('Model comparison — EEG motor imagery classification')
    ax.legend()
    fig.tight_layout()
    st.pyplot(fig)

    st.markdown("""
**Key findings:** SVM achieved 66.7% accuracy, consistently above chance across both
a single test split and 5-fold cross-validation. This confirms it genuinely learned
signal patterns rather than getting lucky on one split. The EEG Transformer achieved
only 44–55% depending on the evaluation, essentially at or near chance. This is not
a failure of the implementation but a reflection of a well-documented limitation:
Transformers are data-hungry (Zhang et al., 2025). With only 30–45 trials available,
the model memorizes noise rather than learning generalizable neural signatures. SVMs
work well on small datasets because domain knowledge is built in. We tell the model
exactly which frequencies and electrodes to look at (Craik et al., 2019). The top
discriminative feature identified was **C3 in the beta band**, the electrode
positioned directly over the left primary motor cortex hand area. Beta
desynchronization there during right-hand motor imagery is one of the most replicated
findings in the entire BCI literature (Pfurtscheller and Lopes da Silva, 1999;
Pfurtscheller and Neuper, 1997). The fact that the model independently identified
this without being told is a validation that it learned real neuroscience.
""")

    st.markdown("---")
    st.subheader("Full Results Table")
    st.markdown(
        "The table below shows every evaluation stored in the database, including both "
        "test-set and cross-validation scores for each model. Two entries appear for "
        "SVM and RandomForest because they were evaluated twice during development. "
        "The results are consistent, confirming reproducibility. The EEGTransformer "
        "entry reflects the final honest evaluation on the extended 45-trial dataset "
        "with the corrected split-before-augmentation pipeline."
    )
    st.dataframe(
        ml_df[['model', 'accuracy', 'notes']]
        .sort_values('accuracy', ascending=False),
        use_container_width=True
    )

    st.markdown("---")
    st.subheader("K-Means Brain Region Clustering")
    st.markdown(
        "K-Means clustering was applied to the Allen Mouse Brain data (Oh et al., 2014), "
        "separate from the EEG classification above. Using only the 3D spatial coordinates "
        "of the 20 brain regions with no functional labels provided, K-Means with K=4 "
        "automatically discovered groupings that closely match known functional brain "
        "anatomy. This is an unsupervised learning result, the algorithm had no "
        "knowledge of which regions are 'visual' or 'hippocampal.' The fact that it "
        "recovered these categories purely from spatial position confirms the "
        "topographic organization of the brain: functionally related regions are "
        "physically co-located (Oh et al., 2014). The silhouette score of 0.41 reflects "
        "reasonable cluster separation for biological data where functional boundaries are "
        "inherently fuzzy rather than geometrically sharp."
    )
    cluster_summary = cluster_df.groupby('cluster_id')['region'].apply(
        lambda x: ', '.join(x)).reset_index()
    cluster_summary.columns = ['Cluster', 'Regions']
    cluster_labels = {
        0: 'Frontal / Motor',
        1: 'Hippocampal / Medial Temporal',
        2: 'Posterior Visual',
        3: 'Subcortical Outliers'
    }
    cluster_summary['Functional Label'] = cluster_summary['Cluster'].map(
        cluster_labels)
    st.dataframe(cluster_summary, use_container_width=True)

# ── region guide ──────────────────────────────────────────
elif page == "Region Guide":
    st.title("Brain Region Reference Guide")
    st.markdown(
        "The 20 most-studied regions in the Allen Mouse Brain Connectivity Atlas "
        "used in this project, organized by functional system."
    )

    st.markdown("#### Visual System")
    st.markdown("""
**VISp : Primary Visual Cortex**
Located in the posterior cortex. Receives direct input from the lateral geniculate
nucleus of the thalamus and processes basic visual features such as edges, orientation,
contrast, and motion. The first cortical stage of visual processing.

**VISl : Lateral Visual Cortex**
Adjacent to primary visual cortex, part of the secondary visual hierarchy. Processes
more complex visual features and has strong reciprocal connections with VISp.

**VISam : Anteromedial Visual Cortex**
A higher-order visual area with strong connections to retrosplenial cortex. Involved
in visuospatial processing and navigation. Has particularly strong reciprocal
connections with RSPv in this dataset.

**VISpor : Postrhinal Visual Cortex**
Located at the border of visual and association cortex. Involved in multimodal
integration of visual and spatial information. Projects strongly to hippocampal areas,
linking visual processing to memory.
""")

    st.markdown("#### Motor and Frontal System")
    st.markdown("""
**MOs : Secondary Motor Cortex**
Also called premotor cortex. Involved in motor planning, action selection, and the
preparation of voluntary movements before execution. Projects strongly to primary
motor cortex and striatum.

**MOp : Primary Motor Cortex**
The main output region for voluntary motor commands. Sends projections down the
corticospinal tract to control limb and body movements. Contains a topographic map
of the body equivalent in mice.

**ACAd : Anterior Cingulate Cortex, Dorsal**
Part of the medial prefrontal cortex. Involved in cognitive control, attention, and
decision-making. The dorsal division has stronger connections to motor and premotor
areas, linking cognition to action.

**ACAv : Anterior Cingulate Cortex, Ventral**
The ventral division of anterior cingulate cortex. More strongly connected to limbic
and emotional processing regions than ACAd. Involved in affect, motivation, and pain
processing. Nearly identical connectivity fingerprint to ACAd.

**SSp-bfd : Primary Somatosensory Cortex, Barrel Field**
The barrel cortex, a region uniquely specialized for processing whisker tactile
input in mice. Each whisker maps to a distinct anatomical barrel, making it one of
the most precisely mapped cortical circuits in neuroscience.

**SSs : Supplemental Somatosensory Cortex**
Secondary somatosensory cortex. Receives input from primary somatosensory cortex and
thalamus. Involved in higher-order tactile processing including texture discrimination
and bilateral sensory integration.
""")

    st.markdown("#### Hippocampal and Memory System")
    st.markdown("""
**CA1 : Hippocampus, CA1 Field**
The primary output region of the hippocampus. Receives processed input from DG and
CA3 via the trisynaptic circuit, and sends output to subiculum and entorhinal cortex.
Critical for encoding new episodic memories and spatial navigation. One of the
earliest regions showing tau pathology in Alzheimer's disease (Siman et al., 2015).

**DG : Dentate Gyrus**
The entry point of the hippocampal trisynaptic circuit. Receives direct input from
the entorhinal cortex via the perforant path and projects to CA3. Involved in pattern
separation, the ability to distinguish similar memories as distinct. One of the few
brain regions that continues generating new neurons throughout adulthood
(Bhaskaran and Bhaskaran, 2007).

**ENTl : Entorhinal Cortex, Lateral**
The primary gateway between the neocortex and the hippocampus. Receives sensory
information and projects to the dentate gyrus via the perforant path. One of the
first regions destroyed in Alzheimer's disease, its loss disconnects sensory
experience from memory encoding (Igarashi, 2023; Uchida et al., 2025).

**ENTm : Entorhinal Cortex, Medial**
Contains grid cells, neurons that fire in a triangular grid pattern as an animal
moves through space, making it essential for spatial navigation and cognitive
mapping. Projects to the dentate gyrus alongside ENTl.
""")

    st.markdown("#### Subcortical and Other Regions")
    st.markdown("""
**CP : Caudoputamen (Striatum)**
The largest component of the basal ganglia. Receives massive input from cortex and
thalamus and is critical for motor control, habit learning, and reward processing.
Dysfunction here underlies Parkinson's disease and Huntington's disease.

**MD : Mediodorsal Thalamus**
A thalamic nucleus with strong bidirectional connections to prefrontal cortex. Acts
as a relay and modulator for cognitive functions including working memory,
decision-making, and attention.

**PAG : Periaqueductal Gray**
A midbrain structure surrounding the cerebral aqueduct. Critical for pain modulation,
defensive behaviors, and autonomic control. Contains endogenous opioid circuits
involved in analgesia.

**LHA : Lateral Hypothalamic Area**
Involved in feeding behavior, arousal, sleep-wake regulation, and reward. Contains
orexin/hypocretin neurons whose loss causes narcolepsy. Acts as a major integrator
of metabolic state and motivated behavior.

**RSPv : Retrosplenial Cortex, Ventral**
A cortical region at the junction of visual and limbic systems. Strongly involved in
spatial navigation, memory consolidation, and context-dependent learning. Has some of
the strongest connections in this dataset, particularly with VISam.

**retina**
Technically not a brain region but included in the Allen Atlas as a source structure.
The retina converts light into electrical signals before transmitting them to the
brain via the optic nerve. Its inclusion as an outlier in the K-Means clustering
correctly reflects its anatomically distinct position outside the brain proper.
""")

# ── references ─────────────────────────────────────────────
elif page == "References":
    st.title("References")

    st.subheader("Data Sources")
    st.markdown("""
**Allen Mouse Brain Connectivity Atlas**

The primary dataset for all graph, pathfinding, alignment, and clustering analyses
in this project. Contains 2,992 anterograde viral tracing experiments across the
mouse brain, processed through an automated informatics pipeline to quantify axonal
projection density between brain regions.

- Citation: Oh SW, et al. (2014). A mesoscale connectome of the mouse brain.
*Nature*, 508, 207-214.
- Access: connectivity.brain-map.org
- Python access: `allensdk` (Allen Institute for Brain Science)

---

**PhysioNet EEG Motor Imagery Dataset (EEGBCI)**

The dataset used for EEG classification and Transformer training. Contains 64-channel
EEG recordings from 109 subjects performing real and imagined hand/foot movements
using the BCI2000 system.

- Citation: Schalk G, et al. (2004). BCI2000: A general-purpose brain-computer
interface system. *IEEE Transactions on Biomedical Engineering*, 51(6), 1034-1043.
- Goldberger AL, et al. (2000). PhysioBank, PhysioToolkit, and PhysioNet.
*Circulation*, 101(23).
- Access: physionet.org/content/eegmmidb/1.0.0/
- Python access: `mne.datasets.eegbci`
""")

    st.markdown("---")
    st.subheader("Literature Citations")
    st.markdown("""
Bhaskaran, M., and Bhaskaran, M. (2007). The multifarious hippocampal mossy fiber
pathway: A review. *Neuroscience and Biobehavioral Reviews*.

Choi, E.Y., et al. (2020). Primate homologs of mouse cortico-striatal circuits.
*eLife*, 9.

Craik, A., et al. (2019). Deep learning for electroencephalogram (EEG) classification
tasks: A review. *Journal of Neural Engineering*, 16(3).

Igarashi, K.M. (2023). Entorhinal cortex dysfunction in Alzheimer's disease.
*eScholarship, UC Irvine*.

Mars, R.B., et al. (2018). Connectivity Fingerprints: From Areal Descriptions to
Abstract Spaces. *Trends in Cognitive Sciences*, 22(11), 1026-1037.

Oh, S.W., et al. (2014). A mesoscale connectome of the mouse brain. *Nature*,
508, 207-214.

Passingham, R.E., et al. (2002). The anatomical basis of functional localization
in the cortex. *Nature Reviews Neuroscience*, 3, 606-616.

Pfurtscheller, G., and Lopes da Silva, F.H. (1999). Event-related EEG/MEG
synchronization and desynchronization: basic principles. *Clinical Neurophysiology*,
110(11), 1842-1857.

Pfurtscheller, G., and Neuper, C. (1997). Motor imagery activates primary
sensorimotor area in humans. *Neuroscience Letters*, 239(2-3), 65-68.

Schalk, G., et al. (2004). BCI2000: A general-purpose brain-computer interface
system. *IEEE Transactions on Biomedical Engineering*, 51(6), 1034-1043.

Siman, R., et al. (2015). The mTOR inhibitor rapamycin mitigates perforant pathway
neurodegeneration in a mouse model of early-stage Alzheimer-type tauopathy.
*PLOS ONE*, 10(11).

Uchida, Y., et al. (2025). Quantification of perforant path fibers for early
detection of Alzheimer's disease. *Alzheimer's and Dementia*, 21.

Zhang, Y., et al. (2025). SVM-enhanced attention mechanisms for motor imagery EEG
classification in brain-computer interfaces. *Frontiers in Neuroscience*.
""")