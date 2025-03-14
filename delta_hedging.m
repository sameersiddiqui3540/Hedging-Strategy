S0 = 100; % Starting Price
r = 0.05; % Interest Rate
T = 1; % Call Option Maturity
n = 5; % Exponent in the number of time steps
N = 1000; % Number of Monte Carlo Sample Paths
mu = 0.02; % Asset Return
sigma = 0.2; % Volatility
K = 80; % Option Strike
c_price = blsprice(S0,K,r,T,sigma); % Black-Scholes call price

n_step = power(2, n); % Number Of Time Steps

time_step = T/n_step; %Time Step Size

time_disc = linspace(0, T, n_step+1); % Vector of 33 numbers evenly/linearly spaced numbers between 0 and 1 (both including)

dW = zeros(n_step+1, N);

dW(2:n_step+1, 1:N) = normrnd(0,1, [n_step, N])*sqrt(time_step);

W = cumsum(dW, 1);

time_grid = (time_disc.').*(ones(n_step+1,N)); % discretised time grid

increment =  sigma*W + (mu-0.5*sigma*sigma)*time_grid; % Black-Scholes Price Increment formula

ST = S0*exp(increment); % Black-Scholes Price Path

tm = ((T-time_disc).').*ones(n_step+1, N);

df_mat = exp(-r*time_grid); % discount factor

d1 = ((log((ST(1:n_step,1:N))./(K*exp(-r*tm(1:n_step,1:N))))-0.5*sigma*sigma*tm(1:n_step,1:N))./(sigma*sqrt(tm(1:n_step,1:N))));

delta = normcdf(d1);

price_diff = ST(2:n_step+1, 1:N).*df_mat(2:n_step+1, 1:N) - ST(1:n_step, 1:N).*df_mat(1:n_step, 1:N); % price difference between two consecutive values of discounted asset.

sum_holding = sum(delta.*price_diff, 1); % Final value of holding of the underlying asset.

X = exp(r*T)*sum_holding*c_price;

upayoff = (ST(end,1:N)-K);
payoff = upayoff.*(upayoff>0); % payoff from option exercise
PNL = X - payoff; % Profit and loss over Monte Carlo sample paths

% Plot of ST
plot(ST)

title('Plot of Stock S');
xlabel('Number Of Time Steps');
ylabel('Stock Price');
legend(['Number of Monte Carlo Sample Paths - ' num2str(N)])
