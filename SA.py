import streamlit as st
from transformers import pipeline
import plotly.express as px

@st.experimental_singleton()
def load_sa_model():
    """
    Create the tensorflow session.
    """
    classifier = pipeline('sentiment-analysis')
    return classifier

@st.experimental_singleton()
def load_gc_model():
    """
    Create the tensorflow session.
    """
    classifier = pipeline("zero-shot-classification")
    return classifier

def adv_predict(cls2,txt, options):
    return cls2(txt, options)


def predict(cls1 , txt):
    return cls1.predict([txt])

def main():
    st.title("Sentiment Analysis Demo")
    cls1 = load_sa_model()

    txt = st.text_area('Enter the Text to Analyse')

    clicked = st.button("Analyse")
    if clicked:
        out = predict(cls1,txt)

        st.write("Sentiment: " ,out[0]["label"])
        st.write("Sentiment Score: ", out[0]["score"])
    
    agree = st.checkbox('Advanced Analysis')

    if agree:
        
        cls2 = load_gc_model()
        options = st.multiselect('What are the categories you want to classify your text?', ['Happy', 'Sad', 'Anger', 'Disgust','Surprise', 'Fear'])
        
        if options:

            out1 = adv_predict(cls2,txt, options)
            
            if out1:
                st.text("Probability Distribution of Selected Categories")
                fig = px.pie( values=out1["scores"], names=out1["labels"])
                fig.update_traces(textposition='inside')
                fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
                # fig.show()
                st.plotly_chart(fig, use_container_width=True)
                

            st.write("Selected Categories: " ,out1["labels"])
            st.write("Category Wise Score: ", out1["scores"])
            # st.json(out1)
            
            # if out1:
            #     fig = px.pie( values=out1["scores"], names=out1["labels"])
            #     fig.update_traces(textposition='inside')
            #     fig.update_layout(uniformtext_minsize=12, uniformtext_mode='hide')
            #     # fig.show()
            #     st.plotly_chart(fig, use_container_width=True)

        # st.write('You selected:', options)





if __name__ == "__main__":
    main()
