function [lls, bics, best_k, labels, cparams] = em_bic(data, max_k)
[rows, features] = size(data);

lls = zeros(max_k, 1);
bics = zeros(max_k, 1);
labels = zeros(rows, max_k, max_k);
default = struct('mu', [], 'covar', [], 'prior', 0);
cparams = repmat(default, max_k, max_k);

for k = 1:max_k
    fprintf('Finding BIC for k=%d\n', k);
    pk = (k - 1) + (k * features) + (k * (features * (features + 1)) / 2);
    [il, ic, l, c, ll] = em(data, k, 10);
    bic = ll(end) - ((pk / 2) * log(rows));
    lls(k) = ll(end);
    bics(k) = bic;
    labels(:, 1:k, k) = l;
    cparams(1:k, k) = c;
end

[~, index] = max(bics);

best_k = index;

end

