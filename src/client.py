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
        "Text Extraction": pt.get_text_template().strip(),
        "Table Extraction": pt.get_table_template().strip(),
        "Classify Document": pt.get_class_template().strip(),
        "Title Extraction": pt.get_title_template().strip(),
    }

def update_template():
    new_label = st.session_state["template_select"]
    if new_label != st.session_state.get("template_label", ""):
        st.session_state.template_label = new_label
        st.session_state.original_prompt = template_options[new_label].format(prompt="")
        st.session_state.prompt = st.session_state.original_prompt

template_options = get_template_options()

def render_ui():
    st.title("Ukrainian Document Adapter")
    col1, col2 = st.columns([2, 1])

    
    base_template_labels = list(template_options.keys())

    if "template_label" not in st.session_state:
        st.session_state.template_label = base_template_labels[0]
    if "original_prompt" not in st.session_state:
        st.session_state.original_prompt = template_options[st.session_state.template_label].format(prompt="")
    if "prompt" not in st.session_state:
        st.session_state.prompt = st.session_state.original_prompt

    with col1:

        col11, col12 = st.columns([1, 1])
        with col11:
            dropdown_index = base_template_labels.index(st.session_state.template_label) \
                if st.session_state.template_label in base_template_labels else 0

            selected_label = st.selectbox(
                "Select a prompt template",
                base_template_labels,
                index=dropdown_index,
                key="template_select",
                on_change=update_template
            )

        with col12:
            temperature = st.slider(
                "Temperature",
                min_value=0.01,
                max_value=2.0,
                value=1.0,
                step=0.01,
                key="temp_slider"
            )

        # Prompt input box
        prompt = st.text_area("Prompt", st.session_state.prompt, height=300, key="prompt_textarea")

        # Detect manual edit (custom)
        if prompt != st.session_state.prompt:
            st.session_state.prompt = prompt
            if prompt != st.session_state.original_prompt:
                st.session_state.template_label = "Custom"

        st.markdown(f"**Prompt Source:** `{st.session_state.template_label}`")

        submit_disabled = not prompt.strip()
        submit_clicked = st.button("Submit", disabled=submit_disabled)

        output_placeholder = st.empty()

    with col2:
        uploaded_file = st.file_uploader("Upload Image", type=["jpg", "jpeg", "png"])
        if uploaded_file:
            st.image(uploaded_file, caption="Preview", use_container_width=True)

    selected_template = template_options.get(st.session_state.template_label)

    return prompt, selected_template, uploaded_file, submit_clicked, output_placeholder, temperature



def submit_to_api(file, prompt, template, output_placeholder, temperature):
    try:
        if template:
            formatted_prompt = template.format(prompt=prompt)
        else:
            formatted_prompt = prompt
    except Exception as e:
        output_placeholder.error(f"Prompt formatting failed: {e}")
        return

    api_key, server_url = get_env_vars()
    url = f"http://{server_url}/generate"
    headers = {"Authorization": f"Bearer {api_key}"}
    files = {"file": file}
    data = {
        "prompt": formatted_prompt,
        "temperature": str(temperature),
    }

    try:
        with output_placeholder:
            with st.spinner("Sending request..."):
                response = requests.post(url, files=files, data=data, headers=headers)
                response.raise_for_status()
                result = response.json()

            st.subheader("Model Response")
            st.text_area("Response", result["response"], height=300)
    except requests.exceptions.RequestException as e:
        output_placeholder.error(f"Request failed: {e}")
    except ValueError:
        output_placeholder.error("Invalid JSON response from server.")



def run_app():
    prompt, selected_template, uploaded_file, submit_clicked, output_placeholder, temperature = render_ui()

    if submit_clicked and uploaded_file:
        submit_to_api(uploaded_file, prompt, selected_template, output_placeholder, temperature)
    elif submit_clicked and not uploaded_file:
        output_placeholder.warning("Please upload an image before submitting.")


if __name__ == "__main__":
    run_app()
