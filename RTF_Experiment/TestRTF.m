%%
load QU_KEMAR_anechoic_1m.mat
%%
HRIR_pos45_R = irs.right(:,226);
HRIR_pos45_L = irs.right(:,136);
HRIR_neg45_R = irs.left(:,226);
HRIR_neg45_L = irs.left(:,136);


subplot(2,1,1)
plot(HRIR_pos45_R, 'r')
hold on
plot(HRIR_pos45_L, 'b')
subplot(2,1,2)
plot(HRIR_neg45_R, '-r')
hold on
plot(HRIR_neg45_L, '-b')
%%
HRTF_pos45_R = abs(fft(HRIR_pos45_R));
HRTF_pos45_R = HRTF_pos45_R(1:end/2);
HRTF_pos45_L = abs(fft(HRIR_pos45_L));
HRTF_pos45_L = HRTF_pos45_L(1:end/2);

figure
plot(10*log10(HRTF_pos45_R), 'r')
hold on
plot(10*log10(HRTF_pos45_L), 'b')
%%
RTF = HRTF_pos45_R./HRTF_pos45_L;
RTF = abs((ifftshift(RTF)));
figure
plot(RTF)
% RTF_pos45 = abs(ifft(fft(HRIR_pos45_R)/)fft(HRIR_pos45_L))

