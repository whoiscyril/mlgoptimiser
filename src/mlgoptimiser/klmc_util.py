def generate_config(ntask, cores_per_worker):
    with open('taskfarm.config', 'w') as f:
        f.write('task_start 0 \n')
        f.write('task_end ' + str(ntask-1) + '\n')
        f.write('cpus_per_worker ' + str(cores_per_worker)  + '\n')
        f.write('application gulp')

