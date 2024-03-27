
import os
from tqdm  import tqdm 
from util import Activity_predict
from post_evaluate.activity.model import DeepSTARR

os.environ['CUDA_VISIBLE_DEVICES'] = "0"

if __name__ == "__main__":
    files = ["../../data/real_Sequence_test.txt",\
             "../../data/real_Sequence_train.txt",\
             "../../data/real_Sequence_val.txt"]
    
    # device = "cuda" if torch.cuda.is_available() else "cpu"

    for file_path in files:
        
        file_ = open(file_path, "r")
        contents = file_.readlines()
        sequences = [content.replace("\n", "") for content in contents]

        batch_size = 512
        activates = []
        length = len(sequences)

        activates = Activity_predict(sequences) # index_0: Dev, index_1: HK

        target_file = os.path.join(os.path.dirname(file_path), os.path.basename(file_path).replace("real", "fake_activity"))
        target_write = open(target_file, "w")
        target_write.write("sequence\tDev_DeepSTARR_log2\tHK_DeepSTARR_log2\n")
        for index in range(len(sequences)):
            content = "{}\t{}\t{}\n".format(sequences[index], activates[0][index][0], activates[1][index][0])
            target_write.write(content)
            target_write.flush()
        target_write.close()

