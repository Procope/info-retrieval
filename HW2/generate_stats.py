import subprocess


data_set = 'test'
score = 'relative_P_5'
score_filename = 'precision5'
mus = [500, 1000, 1500]
kernel = ['passage', 'gaussian', 'triangle', 'hamming', 'circle']
info_ret_path = '/home/akashrajkn/Documents/github_projects/'
qrel_path = info_ret_path + 'info-retrieval/HW2/ap_88_89/qrel_{}'.format(data_set)

for k in kernel:
    for mu in mus:
        print('{0} {1}'.format(k, str(mu)))

        run_path = info_ret_path + 'info-retrieval/HW2/runs/plm/plm_{0}_{1}_{2}.run'.format(k, str(mu), data_set)
        command = 'trec_eval -m all_trec -q {0} {1} | grep -E "{2}\s"'.format(qrel_path, run_path, score)

        process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE)
        out, err = process.communicate()
        out = out.decode("utf-8")

        filename = 'plm_{0}_{1}_{2}_{3}.txt'.format(k, str(mu), score_filename, data_set)
        stats_path = info_ret_path + 'info-retrieval/HW2/statistics/plm/{}/'.format(data_set) + filename

        with open(stats_path, 'w+') as f:
            f.write(out)
