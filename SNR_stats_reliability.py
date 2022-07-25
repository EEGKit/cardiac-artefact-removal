# trialSizes = 5:10:size(myData,2);  ;      %resample these many trials
# noExperiment = 1000;                      %repeat the experiment 1000 times
# subjectSizes = [ 5 10 15 20 24];          %resample these many subjects
# for bIdx = 1:noExperiment % simulate a certain number of experiments
#     for s = 1:numel(subjectSizes)
#         clear subjSample
#         subjSample = datasample(myData, subjectSizes(s),1); %sample certain number of subjects
#         for t = 1:numel(trialSizes)
#             y = datasample(subjSample, trialSizes(t), 2); %sample certain number of trials from each subject
#             [~,p(bIdx,s,t)] = ttest(mean(y,2)); %get the p value from this sub-sample (sampled subjects & sampled trials)
#             effectSizes(bIdx,s,t) = mean(mean(y,2))/std(mean(y,2)); %calculate effect size
#             clear y
#         end
#     end
# end
# end
# % plot the data and look at results
# for t = 1:numel(trialSizes)
#     for s = 1:numel(subjectSizes)
#         Probs(s,t) = sum(p(:,s,t) <0.05)/numel(p(:,s,t));
#     end
# end
# figure; hold on; plot(trialSizes,Probs', 'LineWidth',2)
# ylim([0 1.2])
# ylabel('Proportion of significant experiments')
# xlabel('Number of trials')
# legend({'N = 5', 'N = 10', 'N = 15', 'N = 20', 'N = 24'})
#
