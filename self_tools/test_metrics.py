import torch


if __name__ == '__main__':
    congestion_map = torch.ones([256, 256])
    # congestion_map = model.op_collections.ml_congestion_map_op(self.pos[0])

    top_k_list = [0.005, 0.01, 0.02, 0.05]
    # bins_all = params.route_num_bins_x* params.route_num_bins_y
    bins_all = 256*256

    congestion_map_sorted, indices = torch.sort(congestion_map.view(-1, ))

    avg_cong_sum = 0
    for top_k_idx in top_k_list:
        avg_cong_sum += congestion_map_sorted[:int(top_k_idx*bins_all)].sum(dim=0).data.cpu().numpy()/int(top_k_idx*bins_all)/len(top_k_list)

    print('>>>>> eval metrics >>>>>')
    try:
        from prettytable import PrettyTable
    except ModuleNotFoundError:
        print('need to install prettytable')
    
    # final_hpwl = cur_metric.hpwl

    final_hpwl = 5e7

    shpwl = final_hpwl*(1+0.03*100*avg_cong_sum)
    table = PrettyTable(['HPWL','Cong','sHPWL'])
    table.add_row([f'{final_hpwl}',f'{avg_cong_sum}',f'{shpwl}'])
    print(table)
