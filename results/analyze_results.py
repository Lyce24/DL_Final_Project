#model_parameters,regression_loss,classification_accuracy,precision,recall,f1_score,calories_mae,mass_mae,fat_mae,carbs_mae,protein_mae,overall_improvement
#model_parameters,regression_loss,classification_accuracy,precision,recall,f1_score,calories_mae,mass_mae,fat_mae,carbs_mae,protein_mae,overall_improvement

with open("NutriFusionNet_eval_results_bert.csv", "r") as f:
    next(f)
    
    avg_regression_loss = 0
    avg_classification_accuracy = 0
    avg_precision = 0
    avg_recall = 0
    avg_f1_score = 0
    avg_calories_mae = 0
    avg_mass_mae = 0
    avg_fat_mae = 0
    avg_carbs_mae = 0
    avg_protein_mae = 0
    avg_overall_improvement = 0
    total_mse = 0
    
    count = 0
    for line in f:
        line = line.strip().split(",")
        print(line)
        line = [float(x) for x in line[1:]]
        
        avg_regression_loss += line[0]
        avg_classification_accuracy += line[1]
        avg_precision += line[2]
        avg_recall += line[3]
        avg_f1_score += line[4]
        avg_calories_mae += line[5]
        avg_mass_mae += line[6]
        avg_fat_mae += line[7]
        avg_carbs_mae += line[8]
        avg_protein_mae += line[9]
        avg_overall_improvement += line[10]
        
        total_mse += (line[5] + line[6] + line[7] + line[8] + line[9]) / 5
        
        count += 1
        
    avg_regression_loss /= count
    avg_classification_accuracy /= count
    avg_precision /= count
    avg_recall /= count
    avg_f1_score /= count
    avg_calories_mae /= count
    avg_mass_mae /= count
    avg_fat_mae /= count
    avg_carbs_mae /= count
    avg_protein_mae /= count
    avg_overall_improvement /= count
    total_mse /= count
    
    with open("NutriFusionNet_eval_results_bert_avg.csv", "w") as f:
        f.write("model_parameters,regression_loss,classification_accuracy,precision,recall,f1_score,avg_mae,calories_mae,mass_mae,fat_mae,carbs_mae,protein_mae,overall_improvement\n")
        f.write(f"NutriFusionNet_bert,{avg_regression_loss},{avg_classification_accuracy},{avg_precision},{avg_recall},{avg_f1_score},{total_mse},{avg_calories_mae},{avg_mass_mae},{avg_fat_mae},{avg_carbs_mae},{avg_protein_mae},{avg_overall_improvement}\n")
        
    
with open("NutriFusionNet_eval_results_gat_512.csv", "r") as f:
    next(f)
    
    avg_regression_loss = 0
    avg_classification_accuracy = 0
    avg_precision = 0
    avg_recall = 0
    avg_f1_score = 0
    avg_calories_mae = 0
    avg_mass_mae = 0
    avg_fat_mae = 0
    avg_carbs_mae = 0
    avg_protein_mae = 0
    avg_overall_improvement = 0
    total_mse = 0
    
    index = 0
    count = 0
    for line in f:
        if index > 30:
            break
        line = line.strip().split(",")
        print(line)
        line = [float(x) for x in line[1:]]
        
        avg_regression_loss += line[0]
        avg_classification_accuracy += line[1]
        avg_precision += line[2]
        avg_recall += line[3]
        avg_f1_score += line[4]
        avg_calories_mae += line[5]
        avg_mass_mae += line[6]
        avg_fat_mae += line[7]
        avg_carbs_mae += line[8]
        avg_protein_mae += line[9]
        avg_overall_improvement += line[10]
        
        total_mse += (line[5] + line[6] + line[7] + line[8] + line[9]) / 5
        
        count += 1
        index
        
    avg_regression_loss /= count
    avg_classification_accuracy /= count
    avg_precision /= count
    avg_recall /= count
    avg_f1_score /= count
    avg_calories_mae /= count
    avg_mass_mae /= count
    avg_fat_mae /= count
    avg_carbs_mae /= count
    avg_protein_mae /= count
    avg_overall_improvement /= count
    total_mse /= count
    
    with open("NutriFusionNet_eval_results_gat_512_avg.csv", "w") as f:
        f.write("model_parameters,regression_loss,classification_accuracy,precision,recall,f1_score,avg_mae,calories_mae,mass_mae,fat_mae,carbs_mae,protein_mae,overall_improvement\n")
        f.write(f"NutriFusionNet_gat_512,{avg_regression_loss},{avg_classification_accuracy},{avg_precision},{avg_recall},{avg_f1_score},{total_mse},{avg_calories_mae},{avg_mass_mae},{avg_fat_mae},{avg_carbs_mae},{avg_protein_mae},{avg_overall_improvement}\n")
        
    