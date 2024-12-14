import bentoml
import math

class RepositoryModel():
    def __init__(self, MODEL_FOLDER= "model"):
        self.MODEL_FOLDER = MODEL_FOLDER
     

    def save_model(self, name, model, preprocess=False, postprocess=False, metadata=None):
        '''
        Saves a Keras model to the BentoML store with additional metadata and optional preprocessing/postprocessing steps.
        '''
        bento_model=bentoml.keras.save_model(
            name,
            model,
            include_optimizer=True,
            custom_objects={
                "preprocess": preprocess,
                "postprocess": postprocess,
            },
            metadata=metadata
        )
        return bento_model

    def export_model(self, name):
        '''
        Exports a model from the BentoML store to a .bentomodel file, making it portable or sharable.
        '''
        bentoml.models.export_model(
            name,
            self.MODEL_FOLDER + "/" + f"{name}.bentomodel",
        )
    
    def get_best_model(self, metric, optimum=1):
        '''
        Finds the best model in the BentoML store by comparing a specific metric from the model metadata.
        '''
        all_models = bentoml.models.list()
        best_model = None
        optimum_metric = (-math.inf)*optimum
        for model in all_models:
            if model.info.metadata.get(metric) is not None:  
                if model.info.metadata.get(metric)*optimum > optimum_metric*optimum:
                    optimum_metric = model.info.metadata.get(metric)
                    best_model = model
        
        return best_model
    
    def export_best_model(self, metric, optimum=1):
        '''
        Finds the best model based on a metric and exports it to a .bentomodel file for use elsewhere.
        '''
        best_model = self.get_best_model(metric, optimum)
        if best_model:
            self.export_model(metric, best_model.name, self.MODEL_FOLDER+"/" + f"{best_model.name}.bentomodel")
            return best_model
        else:
            print(f"No model with {metric} metric found.")

    def import_model(self, name,folder=None):
        if folder==None:
            folder=self.MODEL_FOLDER
        print(folder+"/" + f"{name}.bentomodel")
        try:
            bentoml.models.import_model(folder +"/"+ f"{name}.bentomodel")
        except bentoml.exceptions.BentoMLException:
            print("Model already exists in the model store - skipping import.")

    def import_load_model(self, name):
        '''
        Imports a .bentomodel file into the BentoML store (if not already there) and loads the model for use.
        '''
        self.import_model(name)
        model = bentoml.keras.load_model(name)
        return model
    
    def get_model(self, name, version=None):
        return bentoml.models.get(name)
    
    def load_model(self, tag):
        return bentoml.keras.load_model(tag)
    
    def update_model_metadata(self, tag, metadata):
        '''
        Add new metric to the metadata
        '''
        bento_model = self.get_model(tag)
        keras_model = self.load_model(tag)
        updated_metadata = bento_model.info.metadata
        print(metadata)
        for key, value in metadata.items():
            updated_metadata[key] = value
        bentoml.models.delete(tag)
        self.save_model(
            bento_model.tag, 
            keras_model, 
            preprocess=bento_model.custom_objects.get("preprocess"), 
            postprocess=bento_model.custom_objects.get("postprocess"), 
            metadata=updated_metadata)
        bentoml.models.push(bento_model.tag)
        
    def pull_model(self, tag):
        bento_model=bentoml.models.pull(tag)

    def comp_list_model(self,check_model,params,model_list):
        '''
        check if the check_model exist in the list of model
        '''
        check_model_json=check_model.to_json()
        for name_model in model_list:
            self.pull_model(name_model)
            self.export_model(name_model[:-17])
            bento_model=self.get_model(name_model)
            model = bentoml.keras.load_model(name_model)
            model_json=model.to_json()
            if model_json==check_model_json:
                if bento_model.info.metadata["seed"]==params["seed"]:
                    if bento_model.info.metadata["lr"]==params["lr"]:
                        if bento_model.info.metadata["epochs"]==params["epochs"]:
                            return model,bento_model.info.metadata
        return None,None

    def get_top_models_by_metric(self, metric_key, optimum=1, top_n=10):
        """
        Retrieve the top N models based on a specific metric in their metadata.

        Args:
        - metric_key (str): The key of the metric to sort by.
        - top_n (int): Number of top models to retrieve.

        Returns:
        - list: Top N models sorted by the metric.
        """
        # List all models in the BentoML store
        models = bentoml.models.list()

        # Extract models with the desired metric
        models_with_metrics = []
        for model in models:
            # Access metadata
            metadata = model.info.metadata

            if metric_key.strip() in metadata:
                models_with_metrics.append({
                    "model": model,
                    "metric_value": metadata[metric_key.strip()],
                })
            else:
                print(f"Model {model.tag} does not have the metric '{metric_key}'.")
    
        # Sort models by the metric value (descending order for higher-is-better metrics)
        sorted_models = sorted(models_with_metrics, key=lambda x: x["metric_value"], reverse=optimum < 0)

        # Return the top N models
        return list(map(lambda m : m.get("model"), sorted_models[:top_n]))


if __name__ == "__main__":
    repository = RepositoryModel()
    best_model = repository.get_best_model("seed")
    print(best_model.tag)
    model = repository.load_model(best_model.tag)
    print(model)
