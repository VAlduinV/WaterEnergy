filename = "C:\Users\prime\WaterEnergy\WaterEnergy\data\VILLAGE.xlsx";
data = readtable(filename);

% Вибираємо потрібні стовпці (числові дані)
selected_data_numeric = data(:, 24:32);

% Запускаємо kmeans на перетворених числових даних
% Оператор розпакування комірок {:} застосовується до змінної selected_data_numeric, 
% яка є таблицею. Цей оператор розпаковує вміст кожної комірки таблиці.
% Оператор об'єднання ',:' використовується для об'єднання вмісту всіх комірок в один масив.
clust = kmeans(selected_data_numeric{:,:}, 9);

% [s,h] = silhouette(___) будує силуети і повертає хендл фігури h на додаток до значень силуетів у s.
% Значення силуету, що повертаються у вигляді вектора n на 1 зі значеннями від -1 до 1. 
% Значення силуету вимірює, наскільки точка подібна до точок свого кластера 
% у порівнянні з точками в інших кластерах. Значення варіюються від -1 до 1. 
% Високе значення силуету вказує на те, що точка добре співпадає зі своїм кластером 
% і погано з іншими кластерами.
[value_silhouette, h] = silhouette(selected_data_numeric{:,:}, clust);

% Пошук центрів кластерів за допомогою субтрактивної кластеризації
% Знаходимо центри кластерів, використовуючи однаковий діапазон впливу для всіх вимірів.
centers = subclust(selected_data_numeric{:,:}, 0.9);

% Припустимо, що хочемо використовувати стовпці 24 і 25 для діаграми розсіювання
scatter(selected_data_numeric{:,"CITY2_NEAR"}, selected_data_numeric{:,"RD_m2_NEAR"});
xlabel('Water NEAR');
ylabel('power NEAR');
title('Диаграмма рассеяния');

% Використання фільтрації для видалення рядків з нулями
% selected_data_numeric{:,:}: це оператор розпакування комірок, який перетворює вашу таблицю 
% selected_data_numeric у звичайний масив числових даних. Ваші дані тепер представлені як матриця, 
% де кожен рядок відповідає рядку з таблиці, а кожний стовпець - стовпцю з таблиці.
% all(selected_data_numeric{:,:} ~= 0, 2): Тут ми порівнюємо кожен елемент матриці з нулем 
% (selected_data_numeric{:,:} ~= 0). Це повертає матрицю тих самих розмірів, що і selected_data_numeric, 
% але зі значеннями true там, де елементи не рівні нулю, і false там, де вони рівні нулю. 
% Функція all потім перевіряє, чи всі елементи в кожному рядку цієї матриці не рівні нулю. 
% Якщо це так, то вона повертає true, інакше - false.
% filtered_data = selected_data_numeric(all(selected_data_numeric{:,:} ~= 0, 2), :);: цей рядок обирає 
% тільки ті рядки з selected_data_numeric, для яких усі значення не рівні нулю. 
% Це забезпечує відбір лише тих даних, де всі значення стовпців не є нульовими.
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