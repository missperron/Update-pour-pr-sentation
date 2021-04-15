%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% energieEolienne.m
% Fonction calculant l'�nergie �olienne produite par la sd3(kW)
% Inputs : - ventMoy (1x24) (vecteur du vent moyen par heure dans une
% journ�e)
% Output : - eolMoy (1x24) (vecteur de la puissance produite par heure)
% Auteurs : Dominic Rivest
% Date de cr�ation : 2021-04-08
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

function eolMoy = energieEolienne(ventMoy)

% Param�tres de l'�olienne
N_eol = 1; % Nombre d'�oliennes
V_c = 2.5; % Vitesse de d�marrage (m/s)
V_r = 12; % Vitesse nominale (m/s)
P_reol = 3; % Puissance � la vitesse nominale (kW)
a = (V_c)/(V_c-V_r); % Param�tre a
b = 1/(V_r-V_c); % Param�tre b

% Initialisation des variables
eolMoy = []; % Puissance produite par heure (kW)

% for i = 1:length(ventMoy)
%     if ventMoy(i) < V_c % Cas en-dessous de la vitesse d'arr�t
%         eolMoy(i) = 0;
%     elseif ventMoy(i) >= V_r % Cas au-dessus de la vitesse nominale
%         eolMoy(i) = 3;
%     else % Cas entre les deux vitesses
%         eolMoy(i) = P_reol*(a*ventMoy(i)^3-b);
%     end
% end
% Version lin�aire
for i = 1:length(ventMoy)
    if ventMoy(i) < V_c % Cas en-dessous de la vitesse d'arr�t
        eolMoy(i) = 0;
    elseif ventMoy(i) >= V_r % Cas au-dessus de la vitesse nominale
        eolMoy(i) = 3;
    else % Cas entre les deux vitesses
        eolMoy(i) = 3/9.5*ventMoy(i)-0.7895;%P_reol/(V_r-V_c)*ventMoy(i)-P_reol/(V_r-V_c)*2.5;
    end
end
eolMoy = eolMoy*N_eol;
end