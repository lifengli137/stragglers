import os  

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
                    dpRankID = int(file_name_parts[2].split('.')[0])  
                    if ppRankID not in mapping:  
                        mapping[ppRankID] = {}
                    mapping[ppRankID][dpRankID] = f"{root.split('/')[-1]}:{cudaDeviceId}"
                    with open(os.path.join(root, file), 'r') as f:  
                        for line in f:  
                            stepID, allReduceTime = line.strip().split()  
                            stepID = int(stepID)  
                            allReduceTime = float(allReduceTime)  
  
                            if stepID not in data:  
                                data[stepID] = {}  
                            if ppRankID not in data[stepID]:  
                                data[stepID][ppRankID] = {}  
                            data[stepID][ppRankID][dpRankID] = allReduceTime  
                except ValueError:  
                    print(f"Warning: Skipping file with unexpected format: {file}")  
    

    sorted_data = dict(sorted(data.items(), key=lambda item: item[0]))  

    data_list = []
    for stepID, stepData in sorted_data.items():
        ppRankID = 0
        pp_list = []
        while ppRankID in stepData:
            dp_list = []
            dpRankID = 0
            while dpRankID in stepData[ppRankID]:
                dp_list.append(stepData[ppRankID][dpRankID])
                dpRankID += 1
            if len(dp_list) > 0: 
                pp_list.append(dp_list)
            ppRankID += 1
        if len(pp_list) > 0:
            data_list.append(pp_list)
        stepID += 1
    
    return mapping, data_list  



def adjust_allreduce_times_based_on_min(data_list):  

    if len(data_list) == 0 or len(data_list[0]) == 0 or len(data_list[0][0]) == 0:
        return []
    m = len(data_list)
    n = len(data_list[0])
    p = len(data_list[0][0])

    adjusted_data_list = [[[0 for _ in range(p)] for _ in range(n)] for _ in range(m)]  
    
    for i in range(m): # per step
        for j in range(n): #per pp
            min_time = min(data_list[i][j])
            for k in range(p):
                adjusted_data_list[i][j][k] = round(data_list[i][j][k] - min_time, 2)
    return adjusted_data_list

def average_allreduce_time(data_list, steps=0):  
    
    if len(data_list) == 0 or len(data_list[0]) == 0:
        return []

    m = len(data_list[0])
    n = len(data_list[0][0])

    avg_data_list = [[0 for _ in range(n)] for _ in range(m)]  
    
    for i in range(m):
        for j in range(n):
            sum = 0
            count = 0
            if steps <= 0 or steps > len(data_list):
                steps = len(data_list)
            for k in range(steps):
                x = len(data_list) - 1 - k
                sum += data_list[x][i][j]
                count += 1
            avg_data_list[i][j] = round(sum/count, 2)

    return avg_data_list

def calculate_pipeline_stage_time(data_list):  

    if len(data_list) == 0 or len(data_list[0]) == 0:
        return []

    m = len(data_list[0])
    n = len(data_list[0][0])

    avg_data_list = [[0 for _ in range(n)] for _ in range(m)]  
    
    for i in range(m):
        for j in range(n):
            sum = 0
            count = 0
            if steps <= 0 or steps > len(data_list):
                steps = len(data_list)
            for k in range(steps):
                x = len(data_list) - 1 - k
                sum += data_list[x][i][j]
                count += 1
            avg_data_list[i][j] = round(sum/count, 2)

    return avg_data_list

def calculate_pipeline_stage_time(data_list):  


    if len(data_list) == 0 or len(data_list[0]) == 0 or len(data_list[0][0]) == 0:
        return []

    m = len(data_list)
    n = len(data_list[0])
    p = len(data_list[0][0])

    pipe_data_list = [[[0 for _ in range(p)] for _ in range(n)] for _ in range(m)]  
    

    for i in range(m):
        for j in range(n):
            for k in range(p):
                if j + 1 < n:
                    pipe_data_list[i][j][k] = round(data_list[i][j + 1][k] - data_list[i][j][k], 2)

    return pipe_data_list
   

directory = '.'  # 当前目录  
mapping, data_list = read_mapping_and_allreduce_time_from_files(directory)  

adjusted_data_list = adjust_allreduce_times_based_on_min(data_list)
avg_allreduce_times = average_allreduce_time(adjusted_data_list, steps=0)

pipe_data_list = calculate_pipeline_stage_time(data_list)
avg_pipe_times = average_allreduce_time(pipe_data_list)




print("**********************GPU kernel Waiting time (The lower the more suspicious)*******************************")

print("", end='\t\t')
for i in range(len(avg_allreduce_times)):
    print(f"PP{i}", end='\t\t')
print("\n")        

for j in range(len(avg_allreduce_times[0])):
    print(f"DP{j}", end='\t\t')
    for i in range(len(avg_allreduce_times)):           
        print(f"{avg_allreduce_times[i][j]:.2f}({mapping[i][j]})", end='\t')
    print("\n")



print("\n\n\n*********************BW+P2P time (The higher the more suspicious)********************************")


print("", end='\t\t')
for i in range(len(avg_pipe_times)):
    print(f"PP{i}", end='\t\t\t')
print("\n")        

for j in range(len(avg_pipe_times[0])):
    print(f"DP{j}", end='\t\t')
    for i in range(len(avg_pipe_times)):           
        print(f"{avg_pipe_times[i][j]:.2f}({mapping[i][j]})", end='\t')
    print("\n")
