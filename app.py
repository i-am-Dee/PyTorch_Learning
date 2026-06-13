import streamlit as st
import torch
import graphviz

# ── Page Config (must be first Streamlit call) ────────────────────────────────
st.set_page_config(
    page_title="PyTorch Autograd Visualizer",
    page_icon="⚡",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ── Helper: works with both old and new Streamlit versions ────────────────────
def html(content: str):
    """Use st.html() if available (Streamlit ≥1.31), else fall back."""
    try:
        st.html(content)
    except AttributeError:
        st.markdown(content, unsafe_allow_html=True)

# ── Global CSS (only for Streamlit native components) ─────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&family=JetBrains+Mono:wght@400;500;700&display=swap');

html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

.stApp {
    background: linear-gradient(160deg, #0D0F14 0%, #0F1520 60%, #0D0F14 100%);
}

/* Sidebar */
[data-testid="stSidebar"] {
    background: #111827 !important;
    border-right: 1px solid #1E2740 !important;
}
[data-testid="stSidebar"] * { color: #C4CCDB !important; }
[data-testid="stSidebar"] h3 {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.65rem !important;
    letter-spacing: 0.18em !important;
    text-transform: uppercase !important;
    color: #4D9EFF !important;
    margin-top: 20px !important;
}

/* Number inputs */
[data-testid="stNumberInput"] input {
    background: #0D0F14 !important;
    border: 1px solid #1E2740 !important;
    border-radius: 8px !important;
    color: #4D9EFF !important;
    font-family: 'JetBrains Mono', monospace !important;
}
[data-testid="stNumberInput"] input:focus {
    border-color: #4D9EFF !important;
    box-shadow: 0 0 0 2px rgba(77,158,255,0.15) !important;
}
[data-testid="stNumberInput"] label {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.75rem !important;
    color: #6B7590 !important;
}

/* Metrics */
[data-testid="metric-container"] {
    background: #161B27 !important;
    border: 1px solid #1E2740 !important;
    border-radius: 12px !important;
    padding: 20px 18px 16px !important;
}
[data-testid="stMetricLabel"] p {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 0.7rem !important;
    color: #6B7590 !important;
}
[data-testid="stMetricValue"] {
    font-family: 'JetBrains Mono', monospace !important;
    font-size: 1.5rem !important;
    font-weight: 700 !important;
    color: #F5F7FA !important;
}

/* Graphviz */
[data-testid="stGraphVizChart"] {
    background: #111827 !important;
    border-radius: 14px !important;
    padding: 20px !important;
    border: 1px solid #1E2740 !important;
}

/* Divider */
hr { border-color: #1E2740 !important; }
</style>
""", unsafe_allow_html=True)


# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:
    html("""
    <div style="padding:20px 4px 8px; font-family:'JetBrains Mono',monospace;">
        <div style="font-size:0.62rem;letter-spacing:0.22em;color:#4D9EFF;text-transform:uppercase;margin-bottom:4px;">
            PyTorch · Autograd
        </div>
        <div style="font-size:1.15rem;font-weight:700;color:#F5F7FA;">
            Parameters
        </div>
    </div>
    """)

    st.markdown("### 📥 Inputs")
    x_val = st.number_input("x  ·  Input value",   value=3.0, step=0.1, format="%.2f")
    w_val = st.number_input("w  ·  Weight",         value=2.0, step=0.1, format="%.2f")
    b_val = st.number_input("b  ·  Bias",           value=1.0, step=0.1, format="%.2f")
    y_val = st.number_input("y  ·  Target (label)", value=10.0, step=0.1, format="%.2f")

    st.divider()

    st.markdown("### 🔁 Direction")
    mode = st.radio(
        "Pass direction",
        ["⟶  Forward Pass", "⟵  Backward Pass"],
        label_visibility="collapsed",
    )
    is_backward = "Backward" in mode

    st.divider()

    st.markdown("### 🏷 Legend")
    html("""
    <div style="font-family:'JetBrains Mono',monospace;font-size:0.74rem;line-height:2.2;color:#6B7590;">
        <span style="color:#52D78A;font-size:1rem;">■</span>&nbsp; Leaf tensor (requires_grad)<br>
        <span style="color:#4D9EFF;font-size:1rem;">■</span>&nbsp; Operation node<br>
        <span style="color:#FF6B6B;font-size:1rem;">■</span>&nbsp; Loss node<br>
        <span style="color:#FF6B6B;">──→</span>&nbsp; Gradient flow
    </div>
    """)

    st.divider()

    st.markdown("### 𝑓 Formula")
    html("""
    <div style="background:#0D0F14;border:1px solid #1E2740;border-left:3px solid #4D9EFF;
                border-radius:8px;padding:14px;font-family:'JetBrains Mono',monospace;
                font-size:0.76rem;color:#8892A4;line-height:2.2;">
        wx &nbsp;&nbsp;&nbsp;= w × x<br>
        z &nbsp;&nbsp;&nbsp;&nbsp;= wx + b<br>
        <span style="color:#FF6B6B;">loss = (z − y)²</span>
    </div>
    """)


# ── PyTorch Computation ───────────────────────────────────────────────────────
x        = torch.tensor([x_val], requires_grad=True)
w        = torch.tensor([w_val], requires_grad=True)
b        = torch.tensor([b_val], requires_grad=True)
y_target = torch.tensor([y_val])

wx   = w * x
z    = wx + b
loss = (z - y_target) ** 2
loss.backward()

dz_val = (2 * (z - y_target)).item()


# ── Hero Header ───────────────────────────────────────────────────────────────
html(f"""
<div style="background:linear-gradient(135deg,#111827 0%,#141d2e 100%);
            border:1px solid #1E2740;border-radius:18px;padding:40px 48px;
            margin-bottom:28px;position:relative;overflow:hidden;">
    <div style="position:absolute;top:-60px;right:-60px;width:280px;height:280px;
                background:radial-gradient(circle,rgba(77,158,255,0.09) 0%,transparent 70%);
                border-radius:50%;pointer-events:none;"></div>
    <div style="position:absolute;bottom:-80px;left:40%;width:220px;height:220px;
                background:radial-gradient(circle,rgba(255,107,107,0.07) 0%,transparent 70%);
                border-radius:50%;pointer-events:none;"></div>
    <span style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
                 letter-spacing:0.22em;color:#4D9EFF;text-transform:uppercase;
                 display:block;margin-bottom:10px;">
        PyTorch Internals · Computational Graph · Dynamic DAG
    </span>
    <h1 style="font-family:'JetBrains Mono',monospace;font-size:2.4rem;font-weight:700;
               color:#F5F7FA;margin:0 0 14px 0;line-height:1.15;letter-spacing:-0.02em;">
        Autograd <span style="color:#4D9EFF;">Visualizer</span>
    </h1>
    <p style="color:#6B7590;font-size:0.92rem;margin:0;max-width:580px;line-height:1.65;">
        Trace how PyTorch constructs a dynamic DAG on the forward pass and propagates
        Vector–Jacobian Products back through the chain rule.
        Change any parameter on the left to see the graph update live.
    </p>
</div>
""")


# ── Pass Banner ───────────────────────────────────────────────────────────────
if not is_backward:
    html(f"""
    <div style="background:rgba(77,158,255,0.07);border:1px solid rgba(77,158,255,0.22);
                border-radius:10px;padding:13px 20px;margin-bottom:20px;">
        <span style="font-size:1.2rem;">⟶</span>
        <span style="color:#4D9EFF;font-weight:600;font-size:0.87rem;margin-left:10px;">Forward Pass</span>
        <span style="color:#6B7590;font-size:0.82rem;margin-left:10px;">
            Values propagate left → right. PyTorch silently records every operation to build the graph.
        </span>
        <span style="float:right;font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#3a4562;">
            z = {z.item():.4f}
        </span>
    </div>
    """)
else:
    html(f"""
    <div style="background:rgba(255,107,107,0.07);border:1px solid rgba(255,107,107,0.22);
                border-radius:10px;padding:13px 20px;margin-bottom:20px;">
        <span style="font-size:1.2rem;">⟵</span>
        <span style="color:#FF6B6B;font-weight:600;font-size:0.87rem;margin-left:10px;">Backward Pass</span>
        <span style="color:#6B7590;font-size:0.82rem;margin-left:10px;">
            Gradients propagate right → left via chain rule. Each node accumulates ∂L/∂(itself).
        </span>
        <span style="float:right;font-family:'JetBrains Mono',monospace;font-size:0.72rem;color:#3a1a1a;">
            dL/dz = {dz_val:.4f}
        </span>
    </div>
    """)


# ── Graphviz Graph ─────────────────────────────────────────────────────────────
dot = graphviz.Digraph(comment="Autograd Graph")
dot.attr(rankdir="LR", bgcolor="#111827", pad="0.7",
         nodesep="0.75", ranksep="1.1", fontname="Helvetica")

_base = {"shape": "box", "style": "filled,rounded",
         "fontname": "Helvetica", "fontsize": "13", "margin": "0.3,0.18"}

if not is_backward:
    leaf = {**_base, "fillcolor": "#0e2219", "color": "#52D78A",
            "penwidth": "2",   "fontcolor": "#52D78A"}
    op   = {**_base, "fillcolor": "#0e1626", "color": "#4D9EFF",
            "penwidth": "1.5", "fontcolor": "#C4CCDB"}
    lo   = {**_base, "fillcolor": "#1e0f0f", "color": "#FF6B6B",
            "penwidth": "2",   "fontcolor": "#FF6B6B"}

    dot.node("x",    f"x\n{x.item():.3f}",                **leaf)
    dot.node("w",    f"w\n{w.item():.3f}",                **leaf)
    dot.node("b",    f"b\n{b.item():.3f}",                **leaf)
    dot.node("wx",   f"w × x\n{wx.item():.3f}",           **op)
    dot.node("z",    f"z = wx + b\n{z.item():.3f}",       **op)
    dot.node("y",    f"target y\n{y_target.item():.3f}",  **leaf)
    dot.node("loss", f"Loss = (z−y)²\n{loss.item():.3f}", **lo)

    dot.edge("w",  "wx",   color="#4D9EFF", penwidth="1.8")
    dot.edge("x",  "wx",   color="#4D9EFF", penwidth="1.8")
    dot.edge("wx", "z",    color="#4D9EFF", penwidth="1.8")
    dot.edge("b",  "z",    color="#4D9EFF", penwidth="1.8")
    dot.edge("z",  "loss", color="#FF6B6B", penwidth="2.2")
    dot.edge("y",  "loss", color="#FF6B6B", penwidth="2.2")
else:
    g_leaf = {**_base, "fillcolor": "#0e2219", "color": "#52D78A",
              "penwidth": "2",   "fontcolor": "#52D78A"}
    g_op   = {**_base, "fillcolor": "#1e0f0f", "color": "#FF6B6B",
              "penwidth": "1.5", "fontcolor": "#C4CCDB"}
    g_loss = {**_base, "fillcolor": "#2a0e0e", "color": "#FF6B6B",
              "penwidth": "2.5", "fontcolor": "#FF6B6B"}

    dot.node("loss", "dL/dLoss\n1.000",                     **g_loss)
    dot.node("z",    f"dL/dz\n{dz_val:.3f}",               **g_op)
    dot.node("wx",   f"dL/d(wx)\n{dz_val:.3f}",            **g_op)
    dot.node("x",    f"dL/dx\n{x.grad.item():.3f}",        **g_leaf)
    dot.node("w",    f"dL/dw\n{w.grad.item():.3f}",        **g_leaf)
    dot.node("b",    f"dL/db\n{b.grad.item():.3f}",        **g_leaf)

    e = {"color": "#FF6B6B", "penwidth": "2.5", "arrowsize": "0.85"}
    dot.edge("loss", "z",  **e)
    dot.edge("z",    "b",  **e)
    dot.edge("z",    "wx", **e)
    dot.edge("wx",   "w",  **e)
    dot.edge("wx",   "x",  **e)

html(f"""
<p style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
          letter-spacing:0.18em;color:#3a4562;text-transform:uppercase;margin:0 0 8px 2px;">
    Computational Graph · {'Forward Values' if not is_backward else 'Gradient Flow'}
</p>
""")
st.graphviz_chart(dot, use_container_width=True)


# ── Tensor Metrics ─────────────────────────────────────────────────────────────
html("""
<p style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
          letter-spacing:0.18em;color:#3a4562;text-transform:uppercase;
          margin:32px 0 10px 2px;">
    Tensor .grad Attributes
</p>
""")

m1, m2, m3, m4 = st.columns(4)
with m1: st.metric("∂L/∂w  ·  w.grad", f"{w.grad.item():.4f}")
with m2: st.metric("∂L/∂x  ·  x.grad", f"{x.grad.item():.4f}")
with m3: st.metric("∂L/∂b  ·  b.grad", f"{b.grad.item():.4f}")
with m4: st.metric("Loss  ·  (z−y)²",  f"{loss.item():.4f}")


# ── Chain Rule Derivation Cards ────────────────────────────────────────────────
html("""
<p style="font-family:'JetBrains Mono',monospace;font-size:0.65rem;
          letter-spacing:0.18em;color:#3a4562;text-transform:uppercase;
          margin:32px 0 10px 2px;">
    Chain Rule Derivation
</p>
""")

c1, c2, c3 = st.columns(3)

cards = [
    {
        "col":    c1,
        "accent": "#52D78A",
        "label":  "∂L / ∂w",
        "steps": [
            "= 2(z − y) · x",
            f"= 2({z.item():.3f} − {y_val:.3f}) · {x_val:.3f}",
            f"= {w.grad.item():.4f}",
        ],
    },
    {
        "col":    c2,
        "accent": "#4D9EFF",
        "label":  "∂L / ∂x",
        "steps": [
            "= 2(z − y) · w",
            f"= 2({z.item():.3f} − {y_val:.3f}) · {w_val:.3f}",
            f"= {x.grad.item():.4f}",
        ],
    },
    {
        "col":    c3,
        "accent": "#FF6B6B",
        "label":  "∂L / ∂b",
        "steps": [
            "= 2(z − y) · 1",
            f"= 2({z.item():.3f} − {y_val:.3f})",
            f"= {b.grad.item():.4f}",
        ],
    },
]

for card in cards:
    steps_html = "".join(
        f'<div style="color:#8892A4;font-size:0.8rem;line-height:1.9;'
        f'font-family:JetBrains Mono,monospace;">{s}</div>'
        for s in card["steps"][:-1]
    )
    final = card["steps"][-1]
    with card["col"]:
        html(f"""
        <div style="background:#111827;border:1px solid #1E2740;
                    border-top:2.5px solid {card['accent']};
                    border-radius:12px;padding:18px 20px;">
            <div style="color:{card['accent']};font-family:JetBrains Mono,monospace;
                        font-size:0.8rem;font-weight:700;letter-spacing:0.05em;
                        margin-bottom:12px;">{card['label']}</div>
            {steps_html}
            <div style="color:#F5F7FA;font-family:JetBrains Mono,monospace;
                        font-size:0.95rem;font-weight:700;margin-top:8px;
                        padding-top:10px;border-top:1px solid #1E2740;
                        letter-spacing:0.03em;">{final}</div>
        </div>
        """)


# ── Conceptual Note ────────────────────────────────────────────────────────────
html("""
<div style="margin-top:32px;background:#111827;border:1px solid #1E2740;
            border-left:3px solid #4D9EFF;border-radius:10px;padding:18px 24px;">
    <span style="font-size:1.2rem;">💡</span>
    <span style="color:#4D9EFF;font-size:0.82rem;font-weight:600;margin-left:8px;">
        How PyTorch Autograd Works
    </span>
    <div style="color:#6B7590;font-size:0.82rem;line-height:1.75;margin-top:8px;">
        Every operation on a
        <code style="background:#0D0F14;padding:1px 5px;border-radius:4px;
                     color:#52D78A;font-size:0.78rem;">requires_grad=True</code>
        tensor appends a
        <strong style="color:#C4CCDB;">grad_fn</strong> node to the DAG.
        Calling
        <code style="background:#0D0F14;padding:1px 5px;border-radius:4px;
                     color:#FF6B6B;font-size:0.78rem;">loss.backward()</code>
        traverses this graph in reverse topological order, multiplying local Jacobians
        via the <strong style="color:#C4CCDB;">chain rule</strong> and accumulating
        results into each leaf's
        <code style="background:#0D0F14;padding:1px 5px;border-radius:4px;
                     color:#52D78A;font-size:0.78rem;">.grad</code> attribute.
    </div>
</div>
""")


# ── Footer ─────────────────────────────────────────────────────────────────────
html("""
<div style="margin-top:48px;padding:18px 0;border-top:1px solid #1E2740;
            display:flex;justify-content:space-between;flex-wrap:wrap;gap:8px;">
    <span style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:#2a3147;">
        PyTorch Autograd Visualizer · Built with Streamlit
    </span>
    <span style="font-family:'JetBrains Mono',monospace;font-size:0.68rem;color:#2a3147;">
        torch.autograd · Dynamic DAG · VJP · Chain Rule
    </span>
</div>
""")