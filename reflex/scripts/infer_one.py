import click
from reflex.models.reflex import Reflex


@click.command()
@click.option('--context', type=str, help='Contextual evidence of relation')
@click.option('--entity', type=str, help='Entity name')
@click.option('--template', type=str, help='Relation template. Example: [X] plays for the [Y] to extract an entity [X] plays for')
@click.option('--model-dir', type=str, default='./roberta_large/', help='Model directory')
@click.option('--model-name', type=str, default='model.pt', help='Model name')
@click.option('--device', type=str, default='cpu', help='Device string to put model on.')
@click.option('--k', type=int, default=16, help='Approximation hyperparameter value')
@click.option('--expand/--no-expand', type=bool, default=False, help='Expand the anchor token')
@click.option('--spacy-model-name', type=str, default='en_core_web_lg', help='Name of spacy model to load')
def run(context, entity, template, model_dir, model_name, device, k, expand, spacy_model_name):
    reflex = Reflex(model_dir, model_name, device, k, spacy_model_name)
    prediction = reflex.predict_one(context, entity, template, expand=expand)[0]
    print(prediction)

if __name__ == '__main__':
    run()

