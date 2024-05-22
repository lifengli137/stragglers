import os  
#import pdb; pdb.set_trace()
def read_mapping_and_allreduce_time_from_files(directory):  
    mapping = {}
    data = {}  
    for root, dirs, files in os.walk(directory):  
        for file in files:  
            if file.endswith(".txt"):  
                try:  
                    path_parts = root.split('/')  
                    file_name_parts = file.split('_')  
                    cudaDeviceId = int(file_name_parts[0])  
                    ppRankID = int(file_name_parts[1])  
                    tpRankID = int(file_name_parts[2]) 
                    dpRankID = int(file_name_parts[3].split('.')[0])  
                    if ppRankID not in mapping:  
                        mapping[ppRankID] = {}
                    if tpRankID not in mapping[ppRankID]:  
                        mapping[ppRankID][tpRankID] = {}
                    mapping[ppRankID][tpRankID][dpRankID] = f"{root.split('/')[-1]}:{cudaDeviceId}"
                    with open(os.path.join(root, file), 'r') as f:  
                        for line in f:  
                            stepID, allReduceTime = line.strip().split()  
                            stepID = int(stepID)  
                            allReduceTime = float(allReduceTime)  
  
                            if stepID not in data:  
                                data[stepID] = {}  
                            if ppRankID not in data[stepID]:  
                                data[stepID][ppRankID] = {}  
                            if tpRankID not in data[stepID][ppRankID]:
                                data[stepID][ppRankID][tpRankID] = {}
                            data[stepID][ppRankID][tpRankID][dpRankID] = allReduceTime  
                except ValueError:  
                    print(f"Warning: Skipping file with unexpected format: {file}")  
    

    sorted_data = dict(sorted(data.items(), key=lambda item: item[0]))  

    data_list = []
    for stepID, stepData in sorted_data.items():
        ppRankID = 0
        pp_list = []
        while ppRankID in stepData:
            tp_list = []
            tpRankID = 0
            while tpRankID in stepData[ppRankID]:
                dp_list = []
                dpRankID = 0
                while dpRankID in stepData[ppRankID][tpRankID]:
                    dp_list.append(stepData[ppRankID][tpRankID][dpRankID])
                    dpRankID += 1
                if len(dp_list) > 0: 
                    tp_list.append(dp_list)

                tpRankID += 1
            if len(tp_list) > 0:
                pp_list.append(tp_list)

            ppRankID += 1
        if len(pp_list) > 0:
            data_list.append(pp_list)
        stepID += 1
    
    del data_list[-1]
    
    return mapping, data_list  




def average_time(data_list, steps=0):  
    
    if len(data_list) == 0 or len(data_list[0]) == 0 or len(data_list[0][0]) == 0 or len(data_list[0][0][0]) == 0:
        return []

    n = len(data_list[0])
    p = len(data_list[0][0])
    q = len(data_list[0][0][0])

    avg_data_list = [[[0 for _ in range(q)] for _ in range(p)] for _ in range(n)]  
    
    for j in range(n):
        for k in range(p):
            for l in range(q):
                sum = 0
                count = 0
                if steps <= 0 or steps > len(data_list):
                    steps = len(data_list)
                for i in range(steps):
                    x = len(data_list) - 1 - i
                    sum += data_list[x][j][k][l]
                    count += 1
                avg_data_list[j][k][l] = round(sum/count, 2)

    return avg_data_list




STEPS = 10
directory = '.'   
mapping, data_list = read_mapping_and_allreduce_time_from_files(directory) 

averaged_data_list = average_time(data_list, steps=STEPS)


colors = [  
    "\33[100m",
    "\33[101m",
    "\33[102m",
    "\33[104m",
    "\33[105m",
    "\33[106m",
    "\33[93m", 
    "\33[5m"   
]  

print("**********************AllReduce GPU kernel time *******************************")

print("", end='\t\t')
for j in range(len(averaged_data_list)):
    print(f"PP{j}", end='\t\t')
print("\n")        

for k in range(len(averaged_data_list[0])):
    print(f"TP{k}\n")
    for l in range(len(averaged_data_list[0][0])):
        print(f"DP{l}", end='\t\t')
        for j in range(len(averaged_data_list)):           
            print(f"{colors[l]}{averaged_data_list[j][k][l]:.2f}({mapping[j][k][l]})\x1b[0m", end='  ')
        print("\n")


