# dashboard/streamlit_app.py

import streamlit as st
import yaml
import pandas as pd
import plotly.express as px
from typing import Dict, Any

def create_weights_chart(shaping_config: Dict[str, Any]):
    """Creates a Plotly bar chart from the shaping configuration."""
    
    if not shaping_config:
        st.warning("No `shaping_config` data to display.")
        return

    data = []
    for name, params in shaping_config.items():
        weight = params.get('weight', 1.0) # Default to 1.0 if not specified
        data.append({'Component': name, 'Weight': weight})

    df = pd.DataFrame(data)

    fig = px.bar(
        df,
        x='Component',
        y='Weight',
        title="Weights of Reward Components",
        color='Component',
        text_auto='.2f' # Format text on bars to 2 decimal places
    )
    fig.update_layout(
        xaxis_title="Reward Component",
        yaxis_title="Assigned Weight",
        uniformtext_minsize=8,
        uniformtext_mode='hide'
    )
    st.plotly_chart(fig, use_container_width=True)


def main():
    """Main function to run the Streamlit app."""
    st.set_page_config(page_title="RLlama Reward Dashboard", layout="wide")
    
    st.title("🦙 RLlama Reward Configuration Dashboard")
    st.write(
        "This tool helps you visualize your reward component weights. "
        "Upload your `reward_config.yaml` file to see the distribution of weights "
        "defined in your `shaping_config`."
    )

    uploaded_file = st.file_uploader(
        "Upload your reward_config.yaml",
        type=["yaml", "yml"]
    )

    if uploaded_file is not None:
        try:
            config = yaml.safe_load(uploaded_file)
            
            st.success("✅ YAML file loaded successfully!")
            
            col1, col2 = st.columns(2)

            with col1:
                st.header("📊 Reward Weights")
                shaping_config = config.get("shaping_config", {})
                create_weights_chart(shaping_config)

            with col2:
                st.header("📄 Raw Configuration")
                st.json(config)

        except yaml.YAMLError as e:
            st.error(f"Error parsing YAML file: {e}")
        except Exception as e:
            st.error(f"An unexpected error occurred: {e}")
    else:
        st.info("Awaiting YAML file upload...")

if __name__ == "__main__":
    main()