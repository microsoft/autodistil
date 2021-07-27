## Environment

- This code is modified based on the repository developed by Hugging Face: Transformers v2.1.1
- Prepare environment

  > pip install -r requirements.txt

## Data

- Download the GLUE data by running [this script](https://gist.github.com/W4ngatang/60c2bdb54d156a41194446737ce03e2e) and unpack it to directory GLUE_DIR
- TASK_NAME can be one of CoLA, SST-2, MRPC, STS-B, QQP, MNLI, QNLI, RTE.

  > ${GLUE_DIR}/${TASK_NAME}

## Example
***First train model on MNLI (task-agnostic). Then finetune it on RTE. Finally evaluate it on RTE.***

### Step 0. Go to the configuration folder

  > cd Code/run_yaml/

### Step 1. Training on MNLI (task-agnostic)
  
  - $$AMLT_DATA_DIR/Data_GLUE/glue_data/{TASK_NAME}/ is data folder
  - $$AMLT_DATA_DIR/Local_models/pretrained_BERTs/BERT_base_uncased/ contains the teacher and student initialization. 
  - Please create model_dir and download the pretrained BERT_base_uncased and put it here
  > Train_NoAssist_SeriesEpochs_NoHard_PreModel_RndSampl.yaml
   
### Step 2. Finetuning on RTE

  - $$AMLT_DATA_DIR/Outputs/glue/MNLI/NoAssist/All_NoAug_NoHardLabel_PreModel/ contains the models trained on MNLI
  - Epochs_{Epochs_TrainMNLI} is the different model trained on MNLI
  - Please create the folder of $$AMLT_DATA_DIR/Outputs/glue/MNLI/NoAssist/All_NoAug_NoHardLabel_PreModel/ and put the output models of Step 1 here
  > Train_finalfinetuning_SpecificSubs_SeriesEpochs_NoAssist_NoHardLabel_PretrainedModel.yaml

### Step 3. Evaluation on RTE

  - $$AMLT_DATA_DIR/Outputs/glue/{TASK_NAME}/NoAssist/All_FINETUNING_NoHardLabel_PreModel/SpecificSubs/ contains the models finetuned on RTE
  - FinetuneEpochs_{Finetune_Epochs}_EpochsMNLI_{Epochs_TrainMNLI}_Sub_{Subs} is the different model finetuned on RTE
  - Please create the folder of $$AMLT_DATA_DIR/Outputs/glue/{TASK_NAME}/NoAssist/All_FINETUNING_NoHardLabel_PreModel/SpecificSubs/ and put the output models of Step 2 here
  > Evaluate_SpecificSubs_NoAssist_NoHardLabel_PretrainedModel.yaml

<!-- 
# Project

> This repo has been populated by an initial template to help get you started. Please
> make sure to update the content to build a great experience for community-building.

As the maintainer of this project, please make a few updates:

- Improving this README.MD file to provide a great experience
- Updating SUPPORT.MD with content about this project's support experience
- Understanding the security reporting process in SECURITY.MD
- Remove this section from the README

## Contributing

This project welcomes contributions and suggestions.  Most contributions require you to agree to a
Contributor License Agreement (CLA) declaring that you have the right to, and actually do, grant us
the rights to use your contribution. For details, visit https://cla.opensource.microsoft.com.

When you submit a pull request, a CLA bot will automatically determine whether you need to provide
a CLA and decorate the PR appropriately (e.g., status check, comment). Simply follow the instructions
provided by the bot. You will only need to do this once across all repos using our CLA.

This project has adopted the [Microsoft Open Source Code of Conduct](https://opensource.microsoft.com/codeofconduct/).
For more information see the [Code of Conduct FAQ](https://opensource.microsoft.com/codeofconduct/faq/) or
contact [opencode@microsoft.com](mailto:opencode@microsoft.com) with any additional questions or comments.

## Trademarks

This project may contain trademarks or logos for projects, products, or services. Authorized use of Microsoft 
trademarks or logos is subject to and must follow 
[Microsoft's Trademark & Brand Guidelines](https://www.microsoft.com/en-us/legal/intellectualproperty/trademarks/usage/general).
Use of Microsoft trademarks or logos in modified versions of this project must not cause confusion or imply Microsoft sponsorship.
Any use of third-party trademarks or logos are subject to those third-party's policies. -->
