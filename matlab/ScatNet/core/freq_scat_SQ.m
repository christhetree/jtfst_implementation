function S_adapt_separa = freq_scat_SQ(U, Wop_freq)
% Created by Changhong Wang

if ~isvector(U)
    U = reshape(U, [size(U,1) 1 size(U,2)]);
end
    S_fr = scat(U, Wop_freq);
    S_fr_1 = [S_fr{1, 1}.signal{:}];
    S_fr_1 = reshape(S_fr_1, [size(S_fr_1,1)*size(S_fr_1,2) size(S_fr_1,3)]);
    S_fr_2 = [S_fr{1, 2}.signal{:}];
    Y_Dom = reshape(S_fr_2, [size(S_fr_2,1)*size(S_fr_2,2) size(S_fr_2,3)]);
    S_adapt_separa = Y_Dom;
end