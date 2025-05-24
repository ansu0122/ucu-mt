import os
import streamlit as st
import requests
import prompt_templates as pt
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(layout="wide")

def get_env_vars():
    api_key = os.getenv("FAST_API_KEY")
    server_url = os.getenv("HOST_SERVER")
    if not api_key or not server_url:
        st.error("Missing environment variables. Check .env for FAST_API_KEY and HOST_SERVER.")
        st.stop()
    return api_key, server_url


def get_template_options():
    return {
        "Text Extraction": pt.get_text_template(),
        "Table Extraction": pt.get_table_template(),
        "Classify Document": pt.get_class_template(),
        "Title Extraction": pt.get_title_template(),
    }


def render_ui():
    st.title("Ukrainian Document Adapter")
    col1, col2 = st.columns([2, 1])  # Left: wider for text, Right: narrower for image

    template_options = get_template_options()

    # --- LEFT: Prompt + Response ---
    with col1:
        selected_label = st.selectbox("Select a prompt template", list(template_options.keys()))
        selected_template = template_options[selected_label]
        prompt = st.text_area("Prompt", selected_template.format(prompt=""), height=200)

        submit_disabled = not prompt.strip()
        submit_clicked = st.button("Submit", disabled=submit_disabled)

        output_placeholder = st.empty()

    with col2:
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            st.image(uploaded_file, caption="Preview", use_container_width=True)

    return prompt, selected_template, uploaded_file, submit_clicked, output_placeholder


def submit_to_api(file, prompt, template, output_placeholder):
    try:
        formatted_prompt = template.format(prompt=prompt)
    except Exception as e:
        output_placeholder.error(f"Prompt formatting failed: {e}")
        return

    api_key, server_url = get_env_vars()
    url = f"http://{server_url}/generate"
    headers = {"Authorization": f"Bearer {api_key}"}
    files = {"file": file}
    data = {"prompt": formatted_prompt}

    try:
        with st.spinner("Sending request..."):
            response = requests.post(url, files=files, data=data, headers=headers)
            response.raise_for_status()
            result = response.json()

        output_placeholder.subheader("Model Response")
        output_placeholder.text_area("Response", result["response"], height=300)

    except requests.exceptions.RequestException as e:
        output_placeholder.error(f"Request failed: {e}")
    except ValueError:
        output_placeholder.error("Invalid JSON response from server.")


def run_app():
    prompt, selected_template, uploaded_file, submit_clicked, output_placeholder = render_ui()

    if submit_clicked and uploaded_file:
        submit_to_api(uploaded_file, prompt, selected_template, output_placeholder)
    elif submit_clicked and not uploaded_file:
        output_placeholder.warning("Please upload an image before submitting.")


if __name__ == "__main__":
    run_app()
