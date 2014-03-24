function [init_labels, init_cparams, labels, cparams, lls] = em(data, k, r)

labels = [];
cparams = [];
init_labels = [];
init_cparams = [];
first = true;

for curr_r = 1:r
    [new_init_labels, new_init_cparams, new_labels, new_cparams, new_lls] = em_once(data, k);
    if (first || lls(end) < new_lls(end))
        first = false;
        init_labels = new_init_labels;
        init_cparams = new_init_cparams;
        labels = new_labels;
        cparams = new_cparams;
        lls = new_lls;
    end
end

end

function [init_labels, init_cparams, labels, cparams, lls] = em_once(data, k)

rows = size(data, 1);

init_cparams = initialize(data, k);
init_labels = zeros(rows, k);
cparams = init_cparams;
labels = init_labels;
lls = [];
i = 0;
first = true;

while true
    i = i + 1;
    labels = expectation(data, k, cparams);
    cparams = maximization(data, k, labels);
    lls(i) = gen_ll(data, cparams);
    if (first == false && (lls(end) - lls(end-1)) < 0.000001)
        fprintf('Converged after %d iterations\n', i);
        break;
    end
    first = false;
end

end

function [inits] = initialize(data,k)

[n, f] = size(data);

default = struct('mu',[],'covar',[],'prior',0);
inits = repmat(default, f, 1);

for curr_k = 1:k
    safe = false;
    while ~safe
        i = randi(n);
        mu = data(i,:);
        safe = true;
        
        for prev_k = 1:curr_k-1
            if (mu == inits(prev_k).mu)
                safe = false;
                break;
            end
        end
    end
    
    covar = eye(f);
    prior = 1/k;
    
    inits(curr_k) = struct('mu', mu, 'covar', covar, 'prior', prior);
end

end

function [z] = expectation(data, k, cparams)
[rows, ~] = size(data);
z = zeros(rows, k);

for curr_k = 1:k
    cparam = cparams(curr_k);
    
    for curr_x = 1:rows
        z(curr_x, curr_k) = gauss_pdf(cparam.mu, cparam.covar, data(curr_x, :)) * cparam.prior;
    end
end
for curr_z = 1:rows
    zsum = sum(z(curr_z, :));
    if (zsum > 0.0)
        z(curr_z, :) = z(curr_z, :) / zsum;
    else
        z(curr_z, :) = 1/k;
    end
end

end

function [params] = maximization(data, k, z)

[rows, features] = size(data);

sum_zj = sum(z, 1);
priors = sum_zj/sum(sum_zj);

default = struct('mu',[],'covar',[],'prior',0);
params = repmat(default, k, 1);

for curr_k = 1:k
    mu = zeros(1, features);
    for curr_x = 1:rows
        mu = mu + (data(curr_x, :) * z(curr_x, curr_k));
    end
    
    mu = mu / sum_zj(curr_k);
    
    sigma = eye(features) * 0.000000001;
    for curr_x = 1:rows
        x_minus_mu = data(curr_x, :) - mu;
        sigma = sigma + (transpose(x_minus_mu) * x_minus_mu) * z(curr_x, curr_k);
    end
    sigma = sigma / sum_zj(curr_k);
    params(curr_k) = struct('mu', mu, 'covar', sigma, 'prior', priors(curr_k));
end

end

function [ll] = gen_ll(data, params)

[rows, ~] = size(data);
k = size(params);

ll = 0.0;

for curr_x = 1:rows
    x_ll = 0.0;
    for curr_k = 1:k
        cparam = params(curr_k);
        x_ll = x_ll + gauss_pdf(cparam.mu, cparam.covar, data(curr_x, :)) * cparam.prior;
    end
    ll = ll + log(x_ll);
end

end

function [val] = gauss_pdf(mu, covar, x)

x_minus_mu = (x - mu);

features = size(x,2);
sqrt_part = 1/(sqrt((power(2*pi, features) * det(covar))));
exp_part = exp(-0.5 * x_minus_mu * inv(covar) * transpose(x_minus_mu));

val = sqrt_part * exp_part;

end