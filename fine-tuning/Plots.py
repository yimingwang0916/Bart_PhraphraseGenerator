from Plots_creater import LV_plot, OT_plot, OT_plot_histogram, result_means, LL_plot, RL_plot, LL_comp_plot, OT_comp_plot


# LV lossPlot
train_loss = np.load('/home/yiwang/Datasets/figure_config/LA/sc/normal/6_32_160_e-5/train_loss.npy')
train_loss = train_loss.tolist()
valu_loss = np.load('/home/yiwang/Datasets/figure_config/LA/Quora/normal/6_64_80_3e-5/valu_loss.npy')
valu_loss = valu_loss.tolist()
#print(len(valu_loss))
Plots_creater.LV_plot(train_loss,valu_loss)

# OT & means Plot
original_model_result = np.load('/home/yiwang/Datasets/figure_config/ft_BART/Quora/normal/6_64_80_3e-5/bert/original_model_result.npy')
original_model_result = original_model_result.tolist()
ft_trained_model_result = np.load('/home/yiwang/Datasets/figure_config/ft_BART/Quora/normal/6_64_80_3e-5/bert/trained_model_result.npy')
ft_trained_model_result = ft_trained_model_result.tolist()
trained_model_result = np.load('/home/yiwang/Datasets/figure_config/ft_BART/Quora/colon/6_64_80_3e-5/bert/trained_model_result.npy')
trained_model_result = trained_model_result.tolist()

trained_model_means = np.load('/home/yiwang/Datasets/figure_config/LA/sc/normal/6_32_160_3e-5/BLEU/trained_model_means.npy')
trained_model_means = trained_model_means.tolist()
original_model_means = np.load('/home/yiwang/Datasets/figure_config/LA/sc/normal/6_32_160_3e-5/BLEU/original_model_means.npy')
original_model_means = original_model_means.tolist()
print(original_model_means)
Plots_creater.OT_plot(original_model_result, trained_model_result)
Plots_creater.OT_plot_histogram(original_model_result,ft_trained_model_result,
                                trained_model_result)
Plots_creater.OT_comp_plot(original_model_result, ft_trained_model_result, trained_model_result)
Plots_creater.result_means(original_model_means, trained_model_means)

#LL $ RL plots
source_len = np.load('/home/yiwang/Datasets/figure_config/LA/sc/normal/6_32_160_e-5/source_len.npy')
source_len = source_len.tolist()
output_len = np.load('/home/yiwang/Datasets/figure_config/LA/sc/normal/6_32_160_e-5/output_len.npy')
output_len = output_len.tolist()
reference_len = np.load('/home/yiwang/Datasets/figure_config/LA/sc/normal/6_32_160_3e-5/reference_len.npy')
reference_len = reference_len.tolist()
ratio_RS = np.load('/home/yiwang/Datasets/figure_config/LA/sc/normal/6_32_160_3e-5/ratio_RS.npy')
ratio_RS = ratio_RS.tolist()
ratio_OS = np.load('/home/yiwang/Datasets/figure_config/LA/sc/normal/6_32_160_3e-5/ratio_OS.npy')
ratio_OS = ratio_OS.tolist()
Plots_creater.LL_plot(output_len, reference_len)
Plots_creater.RL_plot(ratio_OS,ratio_RS)

# comparison
reference_len_1 = np.load('/home/yiwang/Datasets/figure_config/ft_BART/sc/normal/6_32_160_3e-5/reference_len.npy')
reference_len_1 = reference_len_1.tolist()
reference_len_2 = reference_len_1
output_len_1 = np.load('/home/yiwang/Datasets/figure_config/ft_BART/Quora/normal/6_64_80_3e-5/output_len.npy')
output_len_1 = output_len_1.tolist()
output_len_2 = np.load('/home/yiwang/Datasets/figure_config/LA/sc/normal/8_32_160_e-5/output_len.npy')
output_len_2 = output_len_2.tolist()
Plots_creater.LL_comp_plot(output_len_1, output_len_2, reference_len_1, reference_len_2)

original_model_result = np.load('/home/yiwang/Datasets/figure_config/ft_BART/sc/normal/6_32_160_3e-5/BLEU/original_model_result.npy')
original_model_result = original_model_result.tolist()
trained_model_result_1 = np.load('/home/yiwang/Datasets/figure_config/ft_BART/sc/normal/6_32_160_3e-5/BLEU/trained_model_result.npy')
trained_model_result_1 = trained_model_result_1.tolist()
trained_model_result_2 = np.load('/home/yiwang/Datasets/figure_config/LA/sc/normal/8_32_160_e-5/BLEU/trained_model_result.npy')
trained_model_result_2 = trained_model_result_2.tolist()

