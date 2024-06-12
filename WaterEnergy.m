filename = "C:\Users\prime\WaterEnergy\src\data\VILLAGE_ok_Dist2_coord-.xlsx";
data = readtable(filename);

% Вибираємо потрібні �?товпці (чи�?лові дані)
selected_data_numeric = data(:, 24:32);

% Запу�?каємо kmeans на перетворених чи�?лових даних
% Оператор розпакуванн�? комірок {:} за�?то�?овуєть�?�? до змінної selected_data_numeric, 
% �?ка є таблицею. Цей оператор розпаковує вмі�?т кожної комірки таблиці.
% Оператор об'єднанн�? ',:' викори�?товуєть�?�? дл�? об'єднанн�? вмі�?ту в�?іх комірок в один ма�?ив.
clust = kmeans(selected_data_numeric{:,:}, 9);

% [s,h] = silhouette(___) будує �?илуети і повертає хендл фігури h на додаток до значень �?илуетів у s.
% Значенн�? �?илуету, що повертають�?�? у вигл�?ді вектора n на 1 зі значенн�?ми від -1 до 1. 
% Значенн�? �?илуету вимірює, на�?кільки точка подібна до точок �?вого кла�?тера 
% у порівн�?нні з точками в інших кла�?терах. Значенн�? варіюють�?�? від -1 до 1. 
% Ви�?оке значенн�? �?илуету вказує на те, що точка добре �?півпадає зі �?воїм кла�?тером 
% і погано з іншими кла�?терами.
[value_silhouette, h] = silhouette(selected_data_numeric{:,:}, clust);

% Пошук центрів кла�?терів за допомогою �?убтрактивної кла�?теризації
% Знаходимо центри кла�?терів, викори�?товуючи однаковий діапазон впливу дл�? в�?іх вимірів.
centers = subclust(selected_data_numeric{:,:}, 0.9);

% Припу�?тимо, що хочемо викори�?товувати �?товпці 24 і 25 дл�? діаграми роз�?іюванн�?
scatter(selected_data_numeric{:,"CITY2_NEAR"}, selected_data_numeric{:,"RD_m2_NEAR"});
xlabel('Water NEAR');
ylabel('power NEAR');
title('Диаграмма ра�?�?е�?ни�?');

% Викори�?танн�? фільтрації дл�? видаленн�? р�?дків з нул�?ми
% selected_data_numeric{:,:}: це оператор розпакуванн�? комірок, �?кий перетворює вашу таблицю 
% selected_data_numeric у звичайний ма�?ив чи�?лових даних. Ваші дані тепер пред�?тавлені �?к матриц�?, 
% де кожен р�?док відповідає р�?дку з таблиці, а кожний �?товпець - �?товпцю з таблиці.
% all(selected_data_numeric{:,:} ~= 0, 2): Тут ми порівнюємо кожен елемент матриці з нулем 
% (selected_data_numeric{:,:} ~= 0). Це повертає матрицю тих �?амих розмірів, що і selected_data_numeric, 
% але зі значенн�?ми true там, де елементи не рівні нулю, і false там, де вони рівні нулю. 
% Функці�? all потім перевір�?є, чи в�?і елементи в кожному р�?дку цієї матриці не рівні нулю. 
% Якщо це так, то вона повертає true, інакше - false.
% filtered_data = selected_data_numeric(all(selected_data_numeric{:,:} ~= 0, 2), :);: цей р�?док обирає 
% тільки ті р�?дки з selected_data_numeric, дл�? �?ких у�?і значенн�? не рівні нулю. 
% Це забезпечує відбір лише тих даних, де в�?і значенн�? �?товпців не є нульовими.
filtered_data = selected_data_numeric(all(selected_data_numeric{:,:} ~= 0, 2), :);

opts = statset('Display','final');
[idx,C] = kmeans(filtered_data{:,:}, 9, 'Distance','cityblock',...
    'Replicates',5,'Options',opts);

figure;
plot(selected_data_numeric{idx==1,"power_NEAR"}, selected_data_numeric{idx==1,"Water_NEAR"}, 'b.','MarkerSize',12);

hold on;
plot(selected_data_numeric{idx==2,"power_NEAR"}, selected_data_numeric{idx==2,"Water_NEAR"}, 'r.','MarkerSize',12);
%   
% plot(C(:,1),C(:,2),'kx',...
%      'MarkerSize',15,'LineWidth',3) 
legend('Cluster 1','Cluster 2','Cluster 3', 'Centroids',...
       'Location','NW')
title 'Cluster Assignments and Centroids'
hold off