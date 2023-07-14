# Parameter summary

  + "dataset_name": name of the dataset ( cifar10, cifar100, svhn)

  + "dataset_label_frag": proportion of labeled data over total data

  + "dataset_seed": seed for random when seperating data
    
  + "dataset_validation_frag": proportion of validation set over total data
    
  + "model_epoch": number of epoch to train
  
  + "optimizer_learning_rate": learning rate of optimizer
  
  + "optimizer_weight_decay": weight decay of optimizer
  
  + "model_batch_size": batch_size for training testing and validating
  
  + "loss_margin": loss margin for triplet loss
  
  + "logging_file": logging file for the finetune stage
  
  + "checkpoint": checkpoint save path for the finetune stage
  
  + "pseudo_thres": threshold to delete pseudo labels with low probability
  
  + "logging_file_pseudo": log file for retraing with pseudo labels
  
  + "model_path": final model save path

  + "learining_rate_p2": lr phrase 2

  + "weight_decay_p2": weight decay phrase 2

  + "epoch_p2": epoch phrase 2

  + "batch_size_p2": batch size phrase 2

  + "margin_p2":loss margin phrase 2

  + "results": result file path to save pair predictions and labels

  + "classification_report": json file path save classification matric with accuracy and f1