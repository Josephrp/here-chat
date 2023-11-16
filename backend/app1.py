import cohere
import gradio as gr
import os

# Initialize the Cohere client with your API key
api_key = os.getenv('COHERE_API_KEY')
if not api_key:
    raise ValueError("Cohere API key not found in environment variables")

# Initialize the Cohere client with the API key from the environment variable
co = cohere.Client(api_key)

def generate_text(prompt):
    # Generate a response using the Cohere model
    response = co.generate(
        model='746cb80b-3fba-43ec-b14f-1b0f84e9ed4d-ft',
        prompt=prompt
    )
    # Return the generated text
    return response.generations[0].text

# Create a Gradio interface
iface = gr.Interface(
    fn=generate_text,
    inputs="text",
    outputs="text",
    title="Welcome to Tonic's 🌈Here.Chat🗣️😷",
    description="""
            ### How To Use 🌈Here.Chat: 
            🗣️📝Chat with the bot via text 👇🏻📩 Press send . Check out the model [Cohere/CoChat](https://dashboard.cohere.com/playground/generate) and check out the dataset used to train using Cohere: [tonic/easyreddit](https://huggingface.co/datasets/Tonic/EasyReddit) . This bot is trained on randomized reddit for maximum creativity. 
            You can also use 🗣️😷Here.Chat via API below or by cloning this space. 🧬🔬🔍 Simply click here: <a style="display:inline-block" href="https://huggingface.co/spaces/TeamTonic/coherebot?duplicate=true"><img src="https://img.shields.io/badge/-Duplicate%20Space-blue?labelColor=white&style=flat&logo=data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAABAAAAAQCAYAAAAf8/9hAAAAAXNSR0IArs4c6QAAAP5JREFUOE+lk7FqAkEURY+ltunEgFXS2sZGIbXfEPdLlnxJyDdYB62sbbUKpLbVNhyYFzbrrA74YJlh9r079973psed0cvUD4A+4HoCjsA85X0Dfn/RBLBgBDxnQPfAEJgBY+A9gALA4tcbamSzS4xq4FOQAJgCDwV2CPKV8tZAJcAjMMkUe1vX+U+SMhfAJEHasQIWmXNN3abzDwHUrgcRGmYcgKe0bxrblHEB4E/pndMazNpSZGcsZdBlYJcEL9Afo75molJyM2FxmPgmgPqlWNLGfwZGG6UiyEvLzHYDmoPkDDiNm9JR9uboiONcBXrpY1qmgs21x1QwyZcpvxt9NS09PlsPAAAAAElFTkSuQmCC&logoWidth=14" alt="Duplicate Space"></a></h3>
            #### Join us : 
            🌟TeamTonic🌟 is always making cool demos! Join our active builder's🛠️community on 👻Discord: [Discord](https://discord.gg/GWpVpekp) On 🤗Huggingface: [TeamTonic](https://huggingface.co/TeamTonic) & [MultiTransformer](https://huggingface.co/MultiTransformer) On 🌐Github: [Polytonic](https://github.com/tonic-ai) & contribute to 🌟 [PolyGPT](https://github.com/tonic-ai/polygpt-alpha)"
            Add this space as a 🤖discordbot to your server by [clicking here](https://discord.com/oauth2/authorize?client_id=1174639739492646993&scope=bot+applications.commands&permissions=326417525824()    
    """,
    theme='ParityError/Anime'
)

# Launch the app
iface.launch()
