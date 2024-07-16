# ChatGPT-Misinfo-Detection
## Is Truth Relative for Large Language Models? Investigating the Use of Role-playing Prompts on Misinformation Detection Accuracy of ChatGPT

**Michael Robert Haupt<sup>\*</sup>, MA<sup>1,2</sup>, Luning Yang<sup>\*</sup>, BS<sup>2</sup>, Tina Purnat, MSc<sup>3</sup>, Tim K. Mackey, MAS, PhD<sup>2,4,5</sup>**

<sup>1</sup>Department of Cognitive Science, University of California San Diego, 9500 Gilman Dr, La Jolla, CA, 92093, USA  
<sup>2</sup>Global Health Policy & Data Institute, San Diego, CA USA  
<sup>3</sup>TH Chan School of Public Health, Harvard University, Boston, MA, USA  
<sup>4</sup>S-3 Research, San Diego, CA USA  
<sup>5</sup>Global Health Program, Department of Anthropology, University of California, San Diego, CA USA

\*both authors have contributed equally to this research

**Address for Correspondence:** Tim K. Mackey

9500 Gilman Dr., Mail Code: 0505  
La Jolla, CA, 92093, USA  

**Email:** [tmackey@ucsd.edu](mailto:tmackey@ucsd.edu)

## Data Availability
Our raw dataset is located in the *data* folder of this repository and is named *MTurk_Empathy_Data.xlsx*. The result of our 30-iteration experiment is located in the *result* folder and is named *coded_results.xlsx*. We also provide all the code for our experiments in the *src* folder to facilitate reproduction of our results.

## Result Reproduction
To reproduce our results, you should have all the required packages installed. We recommend creating a virtual environment before running the following command:
```
pip install -r requirements.txt
```

Once you have installed all the required packages, you can start running the experiment.

The parameters you can pass in are `api_key`, `output_file`, `model`, `temperature`, `iteration`, and `identity`.

- `api_key`: Your OpenAI key. This *must* be provided for the experiment.
- `model`: The OpenAI model to be used for the experiment. Default: gpt-3.5-turbo-0613 (note that this model will be deprecated on 2024-09-13, as indicated on https://platform.openai.com/docs/deprecations)
- `temperature`: The degree of variability in the model's responses, ranging from 0 to 2. Note that a temperature of 0 means no variability at all. If you pass in the same prompt with a temperature of 0, you will always get the same response. Default: *0.7*
- `iteration`: The number of iterations for the same prompt. Note that if the temperature is set to 0, this will always be reset to 1 regardless of the initial setting. Since there is no variability, it is unnecessary to create more than one iteration. Default: *30*
- `identity`: The identity option for this experiment: 0 for excluding all identities, 1 for including only political identities, and 2 for including identities in all categories. For more information, please refer to the Method section in our paper. Default: 2

Except for `api_key`, you can leave the parameters unspecified, and they will be set to their default values. Below is a sample code for the experiment:
```
python src/generate.py --api_key="Your API Key" --iteration=2 --temperature=1
```

If you wish to run with the exact parameter settings as ours (using the default values), run:
```
python src/generate.py --api_key="Your API Key"
```
*Note*: You won't be able to produce the exact same results even if you use the exact same parameters as ours due to the variability introduced by the temperature parameter.
