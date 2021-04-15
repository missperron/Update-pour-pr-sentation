%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% energieEolienne.m
% Fonction calculant l'énergie éolienne produite par la sd3(kW)
% Inputs : - ventMoy (1x24) (vecteur du vent moyen par heure dans une
% journée)
% Output : - eolMoy (1x24) (vecteur de la puissance produite par heure)
% Auteurs : Dominic Rivest
% Date de création : 2021-04-08
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function eolMoy = energieEolienne(ventMoy)

% Paramètres de l'éolienne
N_eol = 1; % Nombre d'éoliennes
V_c = 2.5; % Vitesse de démarrage (m/s)
V_r = 12; % Vitesse nominale (m/s)
P_reol = 3; % Puissance à la vitesse nominale (kW)
a = (V_c)/(V_c-V_r); % Paramètre a
b = 1/(V_r-V_c); % Paramètre b

% Initialisation des variables
eolMoy = []; % Puissance produite par heure (kW)

% for i = 1:length(ventMoy)
%     if ventMoy(i) < V_c % Cas en-dessous de la vitesse d'arrêt
%         eolMoy(i) = 0;
%     elseif ventMoy(i) >= V_r % Cas au-dessus de la vitesse nominale
%         eolMoy(i) = 3;
%     else % Cas entre les deux vitesses
%         eolMoy(i) = P_reol*(a*ventMoy(i)^3-b);
%     end
% end
% Version linéaire
for i = 1:length(ventMoy)
    if ventMoy(i) < V_c % Cas en-dessous de la vitesse d'arrêt
        eolMoy(i) = 0;
    elseif ventMoy(i) >= V_r % Cas au-dessus de la vitesse nominale
        eolMoy(i) = 3;
    else % Cas entre les deux vitesses
        eolMoy(i) = 3/9.5*ventMoy(i)-0.7895;%P_reol/(V_r-V_c)*ventMoy(i)-P_reol/(V_r-V_c)*2.5;
    end
end
eolMoy = eolMoy*N_eol;
end