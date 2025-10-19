# src/frontend/welcome.py
import streamlit as st

CSS = """
<style>
.fade-in {
  opacity: 0;
  animation: fadeIn 1.2s ease-in forwards;
}
@keyframes fadeIn {
  to { opacity: 1; }
}
/* center hero */
.hero {
  display: flex;
  min-height: 70vh;
  align-items: center;
  justify-content: center;
  text-align: center;
}
.hero h1 {
  font-size: 3rem;
  margin-bottom: 0.6rem;
}
.hero p {
  font-size: 1.1rem;
  color: #7a7a7a;
}
.cta-btn {
  margin-top: 1.4rem;
}
</style>
"""

def render_welcome():
    st.markdown(CSS, unsafe_allow_html=True)
    st.markdown(
        """
        <div class="fade-in hero">
          <div>
            <h1>Welcome to <strong>PortfolioQuant.ai</strong></h1>
            <p>AI-powered portfolio optimization, risk analysis, and backtesting — in one place.</p>
            <div class="cta-btn">
              <button onclick="window.parent.postMessage({type:'launch_app'}, '*')" style="padding:10px 18px;border-radius:10px;border:0;background:#3b82f6;color:white;font-weight:600;cursor:pointer;">🚀 Launch Application</button>
            </div>
          </div>
        </div>
        """,
        unsafe_allow_html=True,
    )

    # Light JS bridge to set session state from the button
    st.components.v1.html("""
    <script>
      window.addEventListener('message', (e) => {
        if (e.data && e.data.type === 'launch_app') {
          const streamlitDoc = window.parent;
          streamlitDoc.dispatchEvent(new Event("streamlit:runApp"));
        }
      });
    </script>
    """, height=0)

    # Streamlit fallback button (if JS is blocked)
    if st.button("Launch Application", type="primary", use_container_width=False):
        st.session_state.page = "app"
        st.experimental_rerun()

    # Ensure page switches if JS bridge fired
    if st.runtime.scriptrunner.script_run_context.is_running_with_streamlit:
        # tiny hack: if previous run clicked JS, user will click fallback as well
        pass
