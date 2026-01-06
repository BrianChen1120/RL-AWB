import matplotlib.pyplot as plt
import pandas as pd
import numpy as np

def fill_last_valid_step(input_csv, output_csv, max_steps=100):
    df = pd.read_csv(input_csv)
    new_rows = []
    for image_idx, group in df.groupby('image_idx'):
        group = group.set_index('step')
        last_step = group.index.max()
        if last_step >= max_steps:
            new_rows.append(group.reset_index())
            continue
        last_row = group.loc[last_step].copy()
        for step in range(last_step + 1, max_steps + 1):
            new_row = last_row.copy()
            new_row['step'] = step
            group.loc[step] = new_row

        new_rows.append(group.reset_index())
    filled_df = pd.concat(new_rows, ignore_index=True)
    filled_df.to_csv(output_csv, index=False)

def evaluate(errors):
# some index should by -1, stupid GPT :(
    errors = np.sort(errors)
    n = len(errors)
    f05 = errors[int(np.floor(0.5 * n)) - 1]
    f025 = errors[int(np.floor(0.25 * n)) - 1]
    f075 = errors[int(np.floor(0.75 * n)) - 1]
    med = np.median(errors)
    men = np.mean(errors)
    trimean = 0.25 * (f025 + 2 * f05 + f075)
    bst25 = np.mean(errors[:int(np.floor(0.25 * n))])
    wst25 = np.mean(errors[int(np.floor(0.75 * n) - 1) :])

    return men, med, trimean, bst25, wst25

def inference_analysis(datetime_str, best_model=True):
    # 讀取推論結果 CSV
    name = 'best_model' if best_model else 'end_model'
    try:
        fill_last_valid_step(f'./{datetime_str}/inference_results_{name}.csv', f'./{datetime_str}/inference_results_{name}.csv', max_steps=100)
        df = pd.read_csv(f'./{datetime_str}/inference_results_{name}.csv')
        txt_file = f'./{datetime_str}/inference_results_{name}.txt'
        csv_file = f'./{datetime_str}/statistic_inference_results_{name}.csv'
        png_file = f'./{datetime_str}/Histogram_{name}.png'
    except:
        fill_last_valid_step(f'./training_result/{datetime_str}/inference_results_{name}.csv', f'./training_result/{datetime_str}/inference_results_{name}.csv', max_steps=100)
        df = pd.read_csv(f'./training_result/{datetime_str}/inference_results_{name}.csv')
        txt_file = f'./training_result/{datetime_str}/inference_results_{name}.txt'
        csv_file = f'./training_result/{datetime_str}/statistic_inference_results_{name}.csv'
        png_file = f'./training_result/{datetime_str}/Histogram_{name}.png'

    # 將 step 展開成寬表，以 image_idx 為索引，step 為欄位
    pivot = df.pivot(index='image_idx', columns='step', values='arr')
    # print(pivot)
    # for idx in range(1, 513):
    #     plt.title(f"Convergence Curve of arr {idx}.png")
    #     plt.ylabel('arr value')
    #     plt.xlabel('steps')
    #     plt.plot(pivot.values[idx-1])
    #     plt.grid(True)
    #     plt.savefig(f'./2025-05-03-00-35-14 (SAC stage1)/Convergence Curve/{idx}.png')
    #     # plt.show()
    #     plt.clf()

    # 如果資料到 step 6，就比較 step 0 和 step 6；否則可改用最後一個 step，例如 pivot.columns.max()
    start = pivot[0]
    # np.save('start_data_new_algo.npy', arr=np.array(start))

    end = pivot[pivot.columns.max()]

    start_increased = start[end > start]
    start_decreased = start[end < start]

    increased = (end > start).sum()
    decreased = (end < start).sum()
    equal = (end == start).sum()

    # 建立 bins
    all_data = np.concatenate([start_increased.values, start_decreased.values])
    bins = np.histogram_bin_edges(all_data, bins=50)

    # 計算 histogram 資料
    increased_counts, _ = np.histogram(start_increased, bins=bins)
    decreased_counts, _ = np.histogram(start_decreased, bins=bins)

    # 中心點 + 寬度
    bin_centers = 0.5 * (bins[:-1] + bins[1:])
    width = bins[1] - bins[0]

    # 畫圖
    plt.figure(figsize=(10, 6))
    plt.bar(bin_centers, increased_counts, width=width * 0.8, label=f'Increased ({increased})', color='blue')
    plt.bar(bin_centers, decreased_counts, width=width * 0.8, bottom=increased_counts, label=f'Decreased ({decreased})',
            color='orange')

    for i in range(len(bin_centers)):
        inc = increased_counts[i]
        dec = decreased_counts[i]
        if inc + dec > 0:  # 只標有資料的 bin
            plt.text(bin_centers[i], inc + dec + 1, f"{inc}/{dec}", ha='center', fontsize=9)

    plt.title('Stacked Histogram of Step 0 arr Values (Increased vs Decreased)')
    plt.xlabel('Start arr value')
    plt.ylabel('Frequency')
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(png_file)
    plt.clf()
    plt.close()
    # plt.show()


    print('================Increased================')
    print(pivot[end > start])

    print('================Decreased================')
    print(pivot[end < start])

    print('================Equal================')
    print(pivot[end == start])

    # plt.figure(figsize=(8, 5))
    # plt.hist(pivot.loc[end == start, 0], bins=20, rwidth=0.8)
    # plt.title('Histogram of Start Values (Only Equal Samples)')
    # plt.xlabel('Start arr value')
    # plt.ylabel('Frequency')
    # plt.grid(True)
    # plt.tight_layout()
    # plt.show()

    print(f"Increased (0→6): {increased}")
    print(f"Decreased (0→6): {decreased}")
    print(f"Equal (0→6): {equal}")

    # 將 step 展開成寬表，以 image_idx 為索引，step 為欄位
    pivot_rep = df.pivot(index='image_idx', columns='step', values='arr_rep')
    # print(pivot_rep)
    # 如果資料到 step 6，就比較 step 0 和 step 6；否則可改用最後一個 step，例如 pivot.columns.max()
    start_rep = pivot_rep[0]
    end_rep = pivot_rep[pivot.columns.max()]

    mean_perf, median_perf, trimean_perf, bst25, wst25 = evaluate(np.array(start))
    print("Start Performance (binary errors) [median, mean, trimean, best25%, worst25%]:",
              median_perf, mean_perf, trimean_perf, bst25, wst25)

    mean_rep, median_rep, trimean_rep, bst25_rep, wst25_rep = evaluate(np.array(start_rep))
    print("Start Performance (rep errors) [median, mean, trimean, best25%, worst25%]:",
              median_rep, mean_rep, trimean_rep, bst25_rep, wst25_rep)


    end_mean_perf, end_median_perf, end_trimean_perf, end_bst25, end_wst25 = evaluate(np.array(end))
    print("End Performance (binary errors) [median, mean, trimean, best25%, worst25%]:",
              end_median_perf, end_mean_perf, end_trimean_perf, end_bst25, end_wst25)

    end_mean_rep, end_median_rep, end_trimean_rep, end_bst25_rep, end_wst25_rep = evaluate(np.array(end_rep))
    print("End Performance (rep errors) [median, mean, trimean, best25%, worst25%]:",
              end_median_rep, end_mean_rep, end_trimean_rep, end_bst25_rep, end_wst25_rep)

    with open(txt_file, 'w') as fp:
        fp.write('================Increased================\n')
        fp.write(pivot[end > start].to_string() + '\n\n\n')
        fp.write('================Decreased================\n')
        fp.write(pivot[end < start].to_string() + '\n\n\n')
        fp.write('================Equal================\n')
        fp.write(pivot[end == start].to_string() + '\n\n\n')
        fp.write(f"Increased: {increased}\n")
        fp.write(f"Decreased: {decreased}\n")
        fp.write(f"Equal: {equal}\n\n\n")
        fp.write("Start Performance (binary errors) [median, mean, trimean, best25%, worst25%]: "
                 f"{median_perf}, {mean_perf}, {trimean_perf}, {bst25}, {wst25}\n")
        fp.write("Start Performance (rep errors) [median, mean, trimean, best25%, worst25%]: "
                 f"{median_rep}, {mean_rep}, {trimean_rep}, {bst25_rep}, {wst25_rep}\n\n")
        fp.write("End Performance (binary errors) [median, mean, trimean, best25%, worst25%]: "
                 f"{end_median_perf}, {end_mean_perf}, {end_trimean_perf}, {end_bst25}, {end_wst25}\n")
        fp.write("End Performance (rep errors) [median, mean, trimean, best25%, worst25%]: "
                 f"{end_median_rep}, {end_mean_rep}, {end_trimean_rep}, {end_bst25_rep}, {end_wst25_rep}\n")

    with open(csv_file, 'w', newline='') as csvfile:
        import csv
        writer = csv.writer(csvfile)
        # 寫入標頭
        writer.writerow(['', 'median', 'mean', 'trimean', 'bst25', 'wst25'])
        # 寫入資料列
        writer.writerow([
            f'{increased}/{decreased}/{equal}',
            f'{end_median_perf:.4f} / {end_median_rep:.4f}',
            f'{end_mean_perf:.4f} / {end_mean_rep:.4f}',
            f'{end_trimean_perf:.4f} / {end_trimean_rep:.4f}',
            f'{end_bst25:.4f} / {end_bst25_rep:.4f}',
            f'{end_wst25:.4f} / {end_wst25_rep:.4f}',
        ])

    return median_perf, mean_perf, end_median_perf, end_mean_perf

if __name__ == '__main__':
    # ['LEVI 2025-09-10-14-42-30 (SAC-stage 1) (p percentage)', 'NCC 2025-09-11-17-41-18 (SAC-stage 1) (p percentage)']
    # inference_analysis('最好的訓練配置+inference在最好初始參數/LEVI 2025-09-10-14-42-30 (SAC-stage 1) (p percentage)/Inference_On_LEVI full', best_model=False)
    inference_analysis('Gehler 2025-12-28-03-44-29 (SAC-stage 1) (p percentage)', best_model=False)


    """
    LEVI2LEVI
    Start Performance (binary errors) [median, mean, trimean, best25%, worst25%]: 3.257851357714906 3.4257534505559653 3.2427206838224323 1.478133195666628 5.77175136519003
    Start Performance (rep errors) [median, mean, trimean, best25%, worst25%]: 5.233211649004662 5.835915007157155 5.343107634458727 2.3823740872411987 10.271971643025912
    End Performance (binary errors) [median, mean, trimean, best25%, worst25%]: 3.184115572463413 3.3941265159328236 3.190078017215453 1.4983103554868518 5.660461887296133
    End Performance (rep errors) [median, mean, trimean, best25%, worst25%]: 5.1740118561850466 5.8027182503699555 5.320649769791732 2.4213591817358684 10.177521526595644
    LEVI2NCC
    Start Performance (binary errors) [median, mean, trimean, best25%, worst25%]: 2.240025237634131 3.1894527722181345 2.4338105410431687 0.7692434380102104 7.122436054484655
    Start Performance (rep errors) [median, mean, trimean, best25%, worst25%]: 3.1157851674455928 4.24437428160562 3.3504243825654223 1.0568489547520068 9.252167809413486
    End Performance (binary errors) [median, mean, trimean, best25%, worst25%]: 2.131127156355231 3.1092494602831735 2.3515991140546286 0.7336410009762903 7.069215279075693
    End Performance (rep errors) [median, mean, trimean, best25%, worst25%]: 3.065944944920484 4.153802025407245 3.2989060274650077 0.9871993676933327 9.200734750228115
    
    
    NCC2NCC
    Start Performance (binary errors) [median, mean, trimean, best25%, worst25%]: 2.240025237634131 3.1894527722181345 2.4338105410431687 0.7692434380102104 7.122436054484655
    Start Performance (rep errors) [median, mean, trimean, best25%, worst25%]: 3.1157851674455928 4.24437428160562 3.3504243825654223 1.0568489547520068 9.252167809413486
    End Performance (binary errors) [median, mean, trimean, best25%, worst25%]: 2.2056428148349494 3.111770651762561 2.362728955278643 0.7443421136632353 7.01275721026192
    End Performance (rep errors) [median, mean, trimean, best25%, worst25%]: 3.106876184364878 4.151251045287181 3.33759059107037 1.0043901258699233 9.126319323469387
    NCC2LEVI
    Start Performance (binary errors) [median, mean, trimean, best25%, worst25%]: 3.257851357714906 3.4257534505559653 3.2427206838224323 1.478133195666628 5.77175136519003
    Start Performance (rep errors) [median, mean, trimean, best25%, worst25%]: 5.233211649004662 5.835915007157155 5.343107634458727 2.3823740872411987 10.271971643025912
    End Performance (binary errors) [median, mean, trimean, best25%, worst25%]: 3.201443012394927 3.399709373217163 3.2046631113506017 1.5184304477460122 5.662985961483764
    End Performance (rep errors) [median, mean, trimean, best25%, worst25%]: 5.217857459369178 5.821271359623446 5.360862394158246 2.4705945322087244 10.15005678874948
    """

