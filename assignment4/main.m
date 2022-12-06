%% Read in Graph
E = csvread('data/example2.dat');

%% Get matrices
col1 = E(:,1);
col2 = E(:,2);
max_ids = max(max(col1,col2));
As = sparse(col1, col2, 1, max_ids, max_ids); 
A = full(As);

%[va,ea] = eig(A);

D = diag(sum(A));
L = (sqrt(D) ^ (-1/2)) * A * (sqrt(D) ^ (-1/2));

[v, e] = eigs(L, max_ids, 'SA');

%% Plot
tiledlayout(3,2);
c = flipud(gray);

nexttile
imagesc(L)
colormap(c)
title('Sparcity pattern of the Laplacian matrix')

nexttile
plot(sort(diff(diag(e))), 'o')
title('Eigengap')
grid()

nexttile
plot(sort(v(:, 2)))
axis square
title('Fielder vector')
grid()

nexttile
G = graph(A);
plot(G,'layout','force');
title('Graph')

nexttile
imagesc(A)
colormap(c)
colorbar
title('Adjacency matrix (max is 2 for example1 for some reason)')