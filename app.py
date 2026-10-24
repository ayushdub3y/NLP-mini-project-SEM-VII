import datetime as dt
import html
import logging
from pathlib import Path

import plotly.graph_objects as go
import streamlit as st

import counselling
import emotion_model
import journal

LOGGER = logging.getLogger(__name__)

FONT = "Playfair Display, Georgia, serif"
STARTERS = {
    "Stressed about exams": "I have exams next week and I feel like I haven't studied enough. I can't focus and I'm not sleeping well.",
    "Feeling lonely": "Lately I feel disconnected from everyone around me. I'm surrounded by people but I still feel alone.",
    "Good news today": "I got some really good news today and I can't stop smiling. I feel proud of how far I've come.",
    "Can't stop worrying": "My mind keeps racing with worst-case scenarios about the future and I can't switch it off.",
}

st.set_page_config(page_title="Emotion Counsellor", page_icon="◐", layout="centered")
st.markdown(f"<style>{(Path(__file__).parent / 'style.css').read_text(encoding='utf-8')}</style>", unsafe_allow_html=True)

st.session_state.setdefault("history", [])
st.session_state.setdefault("last", None)
st.session_state.setdefault("entry", "")
st.session_state.setdefault("notice", None)


def use_starter(text):
    st.session_state.entry = text
    st.session_state.last = None
    st.session_state.notice = None


def new_entry():
    st.session_state.entry = ""
    st.session_state.last = None
    st.session_state.notice = None


def clear_journal():
    st.session_state.history = []
    new_entry()


def md(markup):
    st.markdown(markup, unsafe_allow_html=True)


def bar_chart(scores):
    items = sorted(scores.items(), key=lambda kv: kv[1])
    top = items[-1][0]
    fig = go.Figure(go.Bar(
        x=[v for _, v in items], y=[k.title() for k, _ in items], orientation="h",
        marker_color=["#ededed" if k == top else "#3a3a3a" for k, _ in items],
        text=[f"{v:.0%}" for _, v in items], textposition="outside", cliponaxis=False,
        textfont=dict(color="#a8a8a8", size=14, family=FONT), hoverinfo="skip",
    ))
    fig.update_layout(
        height=300, margin=dict(l=0, r=40, t=0, b=0), bargap=0.45,
        paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(visible=False, range=[0, max(max(scores.values() if scores else [0.0]) * 1.15, 0.05)]),
        yaxis=dict(tickfont=dict(color="#ededed", size=16, family=FONT), showgrid=False, ticks="", fixedrange=True),
    )
    return fig


def heatmap(history):
    z = [[e.get(em) for e in history] for em in emotion_model.EMOTIONS]
    fig = go.Figure(go.Heatmap(
        z=z, x=[str(i + 1) for i in range(len(history))], y=[em.title() for em in emotion_model.EMOTIONS],
        customdata=[[f"{e['time']:%d %b, %H:%M}" for e in history]] * len(emotion_model.EMOTIONS),
        zmin=0, zmax=1, xgap=3, ygap=3, showscale=False,
        colorscale=[[0, "#141414"], [1, "#f2f2f2"]],
        hovertemplate="%{y}: %{z:.0%}<extra>Entry %{x}, %{customdata}</extra>",
    ))
    fig.update_layout(
        height=340, margin=dict(l=0, r=0, t=0, b=0), paper_bgcolor="rgba(0,0,0,0)", plot_bgcolor="rgba(0,0,0,0)",
        xaxis=dict(type="category", side="bottom", title=dict(text="Entry", font=dict(color="#6f6f6f", size=13, family=FONT)),
                   tickfont=dict(color="#6f6f6f", size=13, family=FONT), ticks="", fixedrange=True), yaxis=dict(tickfont=dict(color="#a8a8a8", size=14, family=FONT), autorange="reversed", ticks=""),
        hoverlabel=dict(bgcolor="#1c1c1c", bordercolor="#3a3a3a", font=dict(family=FONT, color="#ededed")),
    )
    return fig


def render_crisis():
    md(f'<div class="crisis"><h2 class="h-section">You are not alone</h2>'
       '<p>If this describes how you feel, please contact someone you trust who can stay with you. '
       'If you may act on these thoughts or have already hurt yourself, contact local emergency services '
       'or go to the nearest emergency department now.</p>'
       f'<p><a href="{counselling.SUPPORT_DIRECTORY}" target="_blank" rel="noopener noreferrer">'
       'Find support in your country</a></p>'
       f'<p class="hint" style="margin-top:16px">If a counsellor is reviewing this entry, please follow up with the person directly and promptly.</p></div>')


def render_result(last):
    if last["crisis"]:
        render_crisis()
        st.button("Write a new entry", on_click=new_entry)
        return

    result, reply = last["result"], last["reply"]

    scores = result["scores"]
    needs_context = result.get("reading_status") == "needs_context"
    ranked = sorted(scores, key=scores.get, reverse=True)
    second = ranked[1]
    notes = f"with notes of {second}  ·  " if scores[second] >= 0.15 else ""
    if needs_context:
        md('<p class="eyebrow">Your reading</p><h2 class="h-section">How did that feel for you?</h2>')
        st.info("I need more context before suggesting an emotion. Add how you felt about it, "
                "even if your feelings are mixed or hard to name.")
    else:
        md(f'<p class="eyebrow">Your reading</p><h2 class="result-title">{result["top"].title()}</h2>'
           f'<p class="result-sub">{notes}{result["confidence"]:.0%} model score</p>')
    if result.get("uncertain") and not needs_context:
        st.info("This reading is uncertain or mixed. Use the feeling words that fit you best.")
    if result.get("chunks", 1) > 1:
        st.caption("The full entry is included. Scores for longer entries summarize multiple parts of the text.")
    if needs_context:
        with st.expander("Inspect the model's unconfirmed guess"):
            st.caption("This classifier always chooses among seven labels. Its score does not establish your feelings.")
            st.plotly_chart(bar_chart(scores), use_container_width=True, config={"displayModeBar": False})
    else:
        st.plotly_chart(bar_chart(scores), use_container_width=True, config={"displayModeBar": False})

    if result.get("sentences"):
        with st.expander("See it sentence by sentence"):
            rows = "".join(f'<div class="srow"><span class="t">{html.escape(s["sentence"])}</span>'
                           f'<span class="tag">{"Needs context" if s.get("reading_status") == "needs_context" else s["emotion"]}</span></div>' for s in result["sentences"])
            md(rows)

    divergence_js = result.get("divergence_js")
    baseline_info = result.get("baseline")
    entropy_val = result.get("entropy")

    if divergence_js is not None and divergence_js > 0.35 and not needs_context:
        st.info(f"Calibration notice: Notable divergence detected (JS divergence: {divergence_js:.2f}) "
                "between surface vocabulary and contextual semantics.")

    if baseline_info or entropy_val is not None:
        with st.expander("Telemetry & Model Explainability"):
            col1, col2 = st.columns(2)
            with col1:
                if entropy_val is not None:
                    st.metric("Shannon Entropy", f"{entropy_val:.3f}",
                              help="0 = complete certainty, 1 = uniform uncertainty")
            with col2:
                if divergence_js is not None:
                    st.metric("JS Divergence", f"{divergence_js:.3f}",
                              help="Disagreement between statistical baseline and contextual model")
            if baseline_info and baseline_info.get("attributions"):
                st.markdown("**Deterministic Feature Attribution (TF-IDF Coefficients):**")
                chip_list = []
                for item in baseline_info["attributions"]:
                    feat = html.escape(item["feature"])
                    sal = item["salience"]
                    chip_list.append(f'<span class="tag" style="margin-right:6px;" title="Salience: {sal:.3f}">{feat} ({sal:+.2f})</span>')
                chips = " ".join(chip_list)
                md(f'<div style="margin-top:6px; margin-bottom:10px;">{chips}</div>')
                st.caption(f"Statistical baseline top prediction: **{baseline_info['top'].title()}** ({baseline_info['confidence']:.0%})")


    md('<hr class="rule"><p class="eyebrow">A moment of reflection</p>'
       f'<p class="quote">{html.escape(reply["validation"])}</p><div class="spacer"></div>')
    steps = "".join(f"<li>{html.escape(s)}</li>" for s in reply["strategies"])
    md(f'<h2 class="h-section">What may help</h2><ol class="steps">{steps}</ol><div class="spacer"></div>')
    prompts = "".join(f"<li>{html.escape(p)}</li>" for p in reply["prompts"])
    md(f'<h2 class="h-section">Questions to sit with</h2><ul class="plain">{prompts}</ul>')

    md('<div class="spacer"></div>')
    st.button("Write a new entry", on_click=new_entry)


def checkin_tab():
    md('<p class="eyebrow">Daily check-in</p><h1 class="h-display">How are you feeling today?</h1>'
       '<p class="lede">Write in your own words. A few honest sentences are enough. '
       'Entries stay in this session on the server until it ends or you clear the journal. '
       'This app does not save them to disk.</p>')
    st.caption("Write in English. Model scores describe the text and are not a diagnosis. "
               "The first analysis may download the pretrained model.")

    with st.form("checkin", border=False):
        st.text_area("Your entry", key="entry", height=190, label_visibility="collapsed",
                     placeholder="I've been feeling…", max_chars=emotion_model.MAX_ENTRY_CHARS)
        submitted = st.form_submit_button("Reflect on this", type="primary")
    md('<p class="hint">Press Ctrl + Enter to submit.</p>')

    if not st.session_state.last:
        md('<p class="eyebrow" style="margin-top:32px">Not sure where to start?</p>')
        cols = st.columns(2)
        for i, (label, text) in enumerate(STARTERS.items()):
            cols[i % 2].button(label, key=f"starter{i}", on_click=use_starter, args=(text,), use_container_width=True)

    if submitted:
        text = st.session_state.entry.strip()
        st.session_state.last = None
        if not text:
            st.session_state.notice = "Write a sentence or two first, then try again."
        elif counselling.crisis_check(text):
            # Support is independent of model downloads, inference and predicted mood.
            st.session_state.notice = None
            st.session_state.last = {"result": None, "reply": None, "crisis": True}
            st.session_state.history.append({"time": dt.datetime.now().astimezone(), "text": text,
                                             "top": "support requested", "confidence": None, "crisis": True})
        else:
            st.session_state.notice = None
            try:
                with st.spinner("Reading between the lines…"):
                    result = emotion_model.detect(text)
                reply = counselling.respond(result)
            except Exception:
                LOGGER.exception("Emotion analysis failed")
                st.session_state.notice = ("The emotion model could not complete this reading. Your entry is still here. "
                                           "Check that dependencies are installed and that the model is cached or can be downloaded, then retry.")
            else:
                needs_context = result.get("reading_status") == "needs_context"
                st.session_state.last = {"result": result, "reply": reply, "crisis": False}
                st.session_state.history.append({"time": dt.datetime.now().astimezone(), "text": text,
                                                 "top": "needs context" if needs_context else result["top"],
                                                 "confidence": None if needs_context else result["confidence"],
                                                 "uncertain": result["uncertain"], "crisis": False,
                                                 "reading_status": result.get("reading_status", "supported"),
                                                 **({} if needs_context else result["scores"])})

    if st.session_state.notice:
        st.info(st.session_state.notice)
    if st.session_state.last:
        md('<hr class="rule">')
        render_result(st.session_state.last)


def journal_tab():
    history = st.session_state.history
    md('<p class="eyebrow">Mood journal</p><h1 class="h-display">Your patterns over time</h1>')
    if not history:
        md('<p class="lede">Nothing here yet. Complete a check-in and it will appear here, '
           'so you can notice how your feelings shift from day to day.</p>')
        return
    md('<p class="lede">Each column is one entry. Brighter cells mean a higher model score. '
       'Entries needing more context or routed to immediate support have blank cells.</p>')

    flag = counselling.trend_flag(history)
    if flag:
        st.info(flag)
    st.plotly_chart(heatmap(history), use_container_width=True, config={"displayModeBar": False})

    md('<hr class="rule"><h2 class="h-section">Entries</h2>')
    for e in reversed(history):
        with st.expander(f"{e['time']:%d %b, %H:%M}  ·  {e['top'].title()}"):
            st.text(e["text"])

    md('<div class="spacer"></div>')
    left, right = st.columns(2)
    left.download_button("Download as CSV", journal.export_csv(history), "mood_journal.csv", "text/csv", use_container_width=True)
    right.button("Clear journal", on_click=clear_journal, use_container_width=True)


md('<p class="brand">Emotion Counsellor</p>')
tab_checkin, tab_journal = st.tabs(["Check-in", "Journal"])
with tab_checkin:
    checkin_tab()
with tab_journal:
    journal_tab()
md('<hr class="rule"><p class="hint">A supportive tool for reflection. It is not a diagnosis and does not replace a trained counsellor.</p>')
