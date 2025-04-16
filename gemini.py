from google.generativeai import configure, list_models

configure(api_key="AIzaSyA0jbJ3Vf48kIpjzfPIkh2ryIwp4F7QJeE")

models = list_models()
for model in models:
    print(model.name)  # Check available models
