%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
% optiSerre.m
% Programme d'optimisation d'une serre en micror�seau
% Auteurs : Maxime Cecchini, Reda Kaoula, Absa Ndiaye, Marianne Perron,
% Dominic Rivest et Khalil Telhaoui
% Date de cr�ation : 2021-04-01
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

clc
clear
close all
%% Importation des param�tres environnemnentaux
load('DonneesMeteo.mat');

%% Moyenne des donn�es m�t�o horaires par mois
% Cr�ation d'un ensemble de cellules pour les donn�es moyenn�es par heure
param = {}; %(par colonne : Mois, tempMoy, ventMoy, RhMoy, irrMoy, humiAbs, pvMoy, eolMoy, tempSerre, chargeLum, qSerre)
joursMois = [31 28 31 30 31 30 31 31 30 31 30 31];
nomsMois = {'Janvier', 'F�vrier', 'Mars', 'Avril', 'Mai', 'Juin', 'Juillet', 'Ao�t', 'Septembre', 'Octobre', 'Novembre', 'D�cembre'};
solaireTot = [];
eolienTot = [];

for i = 1:12
    % Prise en compte des heures �coul�es dans l'ann�e
    if i == 1
        lignesAccu(i) = 0;
    else
        lignesAccu(i) =  lignesAccu(i-1)+24*joursMois(i-1);
    end
    % Attribution des noms des mois aux cellules de la premi�re colonne
    param{i,1}= nomsMois(i);
    
    % S�lection des donn�es du tableau original selon l'heure du jour et le
    % mois
    tempMoy = [];
    ventMoy = [];
    RHMoy = [];
    irrMoy = [];
    humiAbs = [];
    for j = 1:24 %Indexation pour les heures de la journ�e
        n = 1;
        %Indexation pour les lignes accumul�es
        for k = lignesAccu(i):24:lignesAccu(i)+joursMois(i)*24-24
            tempMoy(j,n) = table2array(DonnesmteoVarennes(j+k,6));
            ventMoy(j,n) = table2array(DonnesmteoVarennes(j+k,3));
            RHMoy(j,n) = table2array(DonnesmteoVarennes(j+k,2));
            irrMoy(j,n) = table2array(DonnesmteoVarennes(j+k,5));
            humiAbs(j,n) = table2array(DonnesmteoVarennes(j+k,8));
            n = n+1;
        end
    end
    % Moyenne mensuelle des donn�es horaires
    param{i,2} = mean(tempMoy'); % Compilation des temp�ratures horaires moyennes (degr�s C)
    param{i,3} = mean(ventMoy'); % Compilation des vents horaires moyennes (m/s)
    param{i,4} = mean(RHMoy'); % Compilation des humidit�s relative horaires moyennes (%)
    param{i,5} = mean(irrMoy'); % Compilation des irradiations horaires moyennes (W/m2?)
    param{i,6} = mean(humiAbs'); % Compilation des humidit�s absolues horaires moy. (g_eau/kg_air)
    param{i,7} = energieSolaire(param{i,5},param{i,2}); % �nergie solaire produite par heure (kW)
    solaireTot(i) = sum(param{i,7});
    param{i,8} = energieEolienne(param{i,3});    % �nergie �olienne produite par heure (kW)
    eolienTot(i) = sum(param{i,8});
    param{i,9} = [16,16,16,16,16,16,19,22,22,22,22,22,22,22,22,22,22,22,19,16,16,16,16,16]; %Temp�rature de la serre
    param{i,10} = [2.2,2.2,2.2,0,0,0,0,0,0,0,0,0,0,0,0,0,0,2.2,2.2,2.2,2.2,2.2,2.2,2.2]; % Charge d'�clairage (kW)
    param{i,11} = chaleurSerre(param{i,5},param{i,2},param{i,8},param{i,10}); % Besoins thermiques de la serre
end

% Puissances �oliennes et solaires horaires annuelles
solaireAnnuel = sum(solaireTot.*joursMois); %�nergie solaire annuelle
eolienAnnuel = sum(eolienTot.*joursMois); %�nergie �olienne annuelle
%% D�finition des param�tres constants du projet
tau = 0.25; % Pas de temps des p�riodes (15 minutes)

% Batterie
s_batt = 24; % Capacit� de la batterie
eta_c_batt = 0.9; % Efficacit� de charge batterie
eta_d_batt = 0.9; % Efficacit� d�charge batterie

% Param�tres divers de puissance
p_fan = 0.1864; % 0,25 hp
p_dh = 0.7457; % 1 hp
%p_humi = 0.7457; % 1 hp

% Param�tres de la serre
p_pompe = 0.200; % 0.200 kW pour la pompe � eau
A_serre = 44; % Aire (m2)
A_v_serre = 0.05*A_serre; % Aire des fen�tres de la serre (m2)
l_serre = 4.4; % Largeur de la serre (m)
L_serre = 10; % Longueur de la serre (m)
h_serre = 4; % Hauteur de la serre (m)
V_serre = l_serre*L_serre*h_serre; % Volume de la serre m3
CO2_min = 800; % (ppm)
CO2_max = 1000; % (ppm)
RH_max = 0.70; % Humidit� relative max
RH_min = 0.50; % Humidit� relative min
P_atm = 101.325; % Pression atmosph�rique (Pa)
rho_a = 1.27; % Densit� de l'air (kg/m3)
cp_a = 1; % Cp de l'air (kJ/kgK)
Q_vf = 18.3; % D�bit d'air volum�trique des fans (m3a/m2_sh)
C_1 = 100; % Constante
C_2 = 1.7001; % Constante (Pa)
C_3 = 7.7835; % Constante (Pa)
C_4 = 1/17.0789; % Constante (k-1)
C_5 = 0.6228; % Constante (kg_2/kg_a)
C_6 = 3600; % Constante
GenCO2_max = 100000; % Concentration de CO2 inject�e par le g�n�rateur (ppm)
CO2_ext = 400; % Concentration de CO2 dans l'air ext�rieur
p_genCO2 = 1; % 1kW pour le generateur
eta_rad = 0.29; % Transmissivit� totale
W_max_irrig = 9.6; % Taux d'irrigation du syst�me d'eau (g/hm2)
W_max_dh = 145; % Taux maximal de d�shumidification (g/hm2)
RH_min = 0.50; % Humidit� relative min
RH_max = 0.70; % Humidit� relative max
p_chauf_max = 10 ; % Puissance �lectrique max du syst�me de chauffage (kW)
p_clim_max = 10 ; % Puissance �lectrique max du syst�me de climatisation (kW)
cop_clim = 2.5; % Coefficient de performance du refroidissement
cop_chauf = 1.7; % Coefficient de performance du chauffage

% Param�tres des plantes
C_res = 1.224*10^(-3); % Coeff de respiration des plantes (g/m2hK)
C_phot = 46.03*10^(-3); % Coeff de photosy. des plantes (g/J)
C_7 = -0.27; % Coeff associ� � la resp des plantes
C_8 = 0.05; % Coeff associ� � la resp. des plantes
W_evap = 125.8; % �vaporation des plantes par heure (g/hm2)


 %% R�solution par mois
sol = {};
fval = {};
for i = 1:12
    %% Entr�e des param�tres du projet du mois
    % R�partition des 24 heures sur les 96 p�riodes
    for k = 1:96
        p_pv_t(k) = param{i,7}(floor((k-1)/4+1)); % Puissance solaire
        p_eol_t(k) = param{i,8}(floor((k-1)/4+1)); % Puissance �olienne
        p_lum_t(k) = param{i,10}(floor((k-1)/4+1)); % Puissance d'�clairage
        q_serre_t(k) = param{i,11}(floor((k-1)/4+1)); % Besoin en chaleur de la serre
        irr_t(k) = param{i,5}(floor((k-1)/4+1)); % Irradiation (W/m2)
        vent_t(k) = param{i,3}(floor((k-1)/4+1)); % Vent (m/s)
        T_serre_t(k) = param{i,9}(floor((k-1)/4+1)); % Temp�rature serre (C)
        T_ext_t(k) = param{i,2}(floor((k-1)/4+1)); % Temp�rature ext�rieure (C)
        humiAbs_t(k) =  param{i,6}(floor((k-1)/4+1)); % Humidit� absolue ext (ge/kga)
    end
    omega_min = (RH_min*C_5*(C_1*(-C_2+C_3*exp(C_4*T_serre_t))))/P_atm; % Humidit� absolue minimale
    omega_max = (RH_max*C_5*(C_1*(-C_2+C_3*exp(C_4*T_serre_t))))/P_atm; % Humidit� absolue maximale
    q_vn_t = 298*(T_serre_t-T_ext_t)/3600*rho_a*cp_a; % Flux de chaleur potentiel des fen�tres
    
    %% D�finition des variables du projet
    % Activation des syst�mes (Fen�tres, Pompe, D�shumi, Chauf, Clim)
    s_tj = optimvar('s_tj',5,96,'Type','integer','LowerBound',0,'UpperBound',1);
    
    % Puissance requise des syst�mes de chauffage et clim
    p_clim_t = optimvar('p_clim_t',1,96,'LowerBound',0);
    p_chauf_t = optimvar('p_chauf_t',1,96,'LowerBound',0);
    
    % Puissance du r�seau externe requise au temps t
    p_ext_t = optimvar('p_ext_t',1,96,'LowerBound',0);
    
    % Concentration de CO2 au temps t
    %CO2_t = optimvar('CO2_t',1,96,'LowerBound',0);
    %GenCO2_t = optimvar('GenCO2_t',1,96,'LowerBound',0); % Input du generateur de CO2
    
    % Variables pour la batterie (Soc, sp, sm)
    SOC_t = optimvar('SOC_t',1,96,'LowerBound',0);
    sp_t = optimvar('sp_t',1,96,'LowerBound',0); %�nergie charg�e au temps t
    sm_t = optimvar('sm_t',1,96,'LowerBound',0); % �nergie d�charg�e au temps t
    p_batt_t = optimvar('p_batt_t',1,96); % Puissance fournie au temps t

    
    % Variables pour l'humidit�
    omega_t = optimvar('omega_t',1,96,'LowerBound',0); % Humidit� de la serre (ge/kga)
    W_dh = optimvar('W_dh',1,96); % Taux de d�shumification de la serre (ge/kga)
    
    %% D�finition de la fonction objectif du projet
    
    prob = optimproblem('ObjectiveSense','min');
    % Objectif de minimisation de la puissance externe
    prob.Objective = sum(tau*p_ext_t);
    
    %% D�finition des contraintes du projet
    
    % Contraintes sur la puissance externe
    prob.Constraints.p_1 = p_batt_t + p_eol_t + p_pv_t + p_ext_t >= p_clim_t + p_chauf_t + p_lum_t + s_tj(2,:)*p_pompe +s_tj(3,:)*p_dh + p_fan;
    
    % Contraintes sur le chauffage et la climatisation
    prob.Constraints.chaleur_1 = p_chauf_t <= s_tj(4,:)*p_chauf_max; % Activation du chauffage
    prob.Constraints.chaleur_2 = p_clim_t <= s_tj(5,:)*p_clim_max; % Activation de la climatisation
    prob.Constraints.chaleur_3 = p_chauf_t >= (q_serre_t+q_vn_t.*s_tj(1,:))/cop_chauf; % Puissance du chauffage
    prob.Constraints.chaleur_4 = p_clim_t >= (q_serre_t+q_vn_t.*s_tj(1,:))/-cop_clim; % Puissance de la climatisation
    
    % Contrainte sur l'activation de la pompe
    prob.Constraints.pomp_1 = s_tj(2,1:4:end) == 1; %Premier 15 minutes de chaque heure
    
    % Contraintes sur la batterie
    prob.Constraints.batt_1 = SOC_t <= s_batt; % Capacit� batterie
    prob.Constraints.batt_2 = SOC_t(2:end) == SOC_t(1:end-1)+eta_c_batt*sp_t(2:end)-sm_t(2:end)/eta_d_batt;% Suivi de la charge
    prob.Constraints.batt_3 = SOC_t(1) == s_batt/2; % Charge initiale
    prob.Constraints.batt_4 = SOC_t(end) >= SOC_t(1); % Charge finale egale a la charge initiale
    prob.Constraints.batt_5 = p_batt_t == (sm_t-sp_t)/tau; % Puissance fournie par la batterie
    prob.Constraints.batt_6 = sm_t(1) == 0; % Dechargement initial nul
    prob.Constraints.batt_7 = sp_t(1) == 0; % Chargement initial nul 
    
    % Contraintes sur l'humidit�
    prob.Constraints.humi_1 = omega_t(1) == (omega_max(1)+omega_min(1))/2; % Humidit� initiale
    prob.Constraints.humi_2 = omega_t(2:end) == omega_t(1:end-1)+tau/(rho_a*V_serre)*(W_evap*A_serre+Q_vf*rho_a.*(humiAbs_t(2:end)-(omega_max(1)+omega_min(1))/2)+vent_t(2:end).*+s_tj(1,2:end).*A_v_serre.*(humiAbs_t(2:end)-(omega_max(1)+omega_min(1))/2)+s_tj(2,2:end)*A_serre*W_max_irrig-A_serre.*W_dh(2:end));
    %+s_tj(3,2:end)*A_serre*W_max_dh); Pas d'humidificateur donc comment�
    prob.Constraints.humi_3 = omega_t >= omega_min; % Humidit� minimale
    prob.Constraints.humi_4 = omega_t <= omega_max; % Humidit� maximale
    prob.Constraints.humi_5 = W_dh <= W_max_dh; % Maximum deshumidification
    prob.Constraints.humi_6 = W_dh <= s_tj(3,:)*W_max_dh; % Activation deshumidification
    
    % Anciennes contraintes sur le CO2 (n'est plus utilis�e)
%     prob.Constraints.CO2_1 = CO2_t(2:end) == CO2_t(1:end-1)+tau/(V_serre)*(GenCO2_t(1,2:end)*A_serre+...
%         C_6.*vent_t(2:end).*A_v_serre*rho_a.*s_tj(7,2:end).*(CO2_ext-(CO2_max+CO2_min)/2)+C_res*A_serre*(C_7+C_8*T_serre_t(2:end)-C_6*C_phot*irr_t(2:end)*eta_rad*A_serre));% �quation du CO2
%     prob.Constraints.CO2_2 = CO2_t(1) == (CO2_min+CO2_max)/2; % Concentration initiale (ppm)
%     prob.Constraints.CO2_3 = CO2_t >= CO2_min; % Concentration min
%     prob.Constraints.CO2_4 = CO2_t <= CO2_max; % Concentration max
%     prob.Constraints.CO2_4 = GenCO2_t <= s_tj(1,:)*GenCO2_max; % Activation g�n�rateur
    
    % R�solution du probl�me du mois
    [sol{i},fval{i}] = solve(prob);
    
    % Correction pour l'activation du chauffage et de la climatisation
    for j = 1:96
        if sol{i}.p_clim_t(j) == 0 % Correction si p_clim est de 0
           sol{i}.s_tj(5,j) = 0;
        end
        if sol{i}.p_chauf_t(j) == 0 % Correction si p_chauf est de 0
           sol{i}.s_tj(4,j) = 0;
        end
    end
end



%% Graphiques

figure()
chargeTotale = []; % Charge �nerg�tique totale (sans les sources d'�nergie)
couleur = {[0.5,0.5,0.5],[1,0.5,0.3],[0.1,0.5,0.8],[0.5,0.8,0.5],[0.2,0.6,0.2],[0.8,0.2,0.2]};
energieEcono = []; % �nergie �conomis�e gr�ce � l'�olienne, au panneau solaire et � la batterie
marqueur = {'-','--'};
temps = [0.25:0.25:24];
for i = 1:12
    subplot(2,3,1)
    plot(temps,sol{i}.p_ext_t,marqueur{mod(i,2)+1},'DisplayName',param{i,1}{1},'Color',couleur{floor((i-1)/2+1)})
    title('Puissance externe par mois en fonction de l''heure')
    hold on
    
    subplot(2,3,2)
    plot([1:24],param{i,7},marqueur{mod(i,2)+1},'DisplayName',param{i,1}{1},'Color',couleur{floor((i-1)/2+1)})
     title('Puissance solaire par mois en fonction de l''heure')
    hold on
    subplot(2,3,3)
    plot([1:24],param{i,8},marqueur{mod(i,2)+1},'DisplayName',param{i,1}{1},'Color',couleur{floor((i-1)/2+1)})
    title('Puissance de l''�olienne par mois en fonction de l''heure')
    hold on
    
    subplot(2,3,4)
    plot(temps,sol{i}.SOC_t,marqueur{mod(i,2)+1},'DisplayName',param{i,1}{1},'Color',couleur{floor((i-1)/2+1)})
    title('Charge de la batterie en fonction de l''heure par mois')
    hold on
    
    chargeTotale(i,:) = sol{i}.p_clim_t + sol{i}.p_chauf_t + p_lum_t + sol{i}.s_tj(2,:)*p_pompe +sol{i}.s_tj(3,:)*p_dh + p_fan;
    
    subplot(2,3,5)
    plot(temps,chargeTotale(i,:),marqueur{mod(i,2)+1},'DisplayName',param{i,1}{1},'Color',couleur{floor((i-1)/2+1)})
    title('Charge �nerg�tique en fonction de l''heure par mois')
    hold on
    
   energieEcono(i,:) = [sum(param{i,8}), sum(param{i,7}), sum(sol{i}.sm_t)]; 
end
subplot(2,3,1)
xlabel('Heure')
ylabel('Puissance externe (kW)')
legend()

subplot(2,3,2)
xlabel('Heure')
ylabel('Puissance solaire (kW)')
legend()

subplot(2,3,3)
xlabel('Heure')
ylabel('Puissance de l"�olienne (kW)')
legend()

subplot(2,3,4)
xlabel('Heure')
ylabel('Charge de la batterie (kWh)')
legend()
ylim([0, s_batt])

subplot(2,3,5)
xlabel('Heure')
ylabel('Charge �nerg�tique (kW)')
legend()

 subplot(2,3,6)
    mois = categorical({'J','F','M','A','M','J','J','A','S','O','N','D'});
    bar(energieEcono,'stacked')
    title('�nergie �conomis�e par les �l�ments en fonction du mois')
    ylabel('�nergie (kWh)')
    xlabel('mois')
    legend('�olienne','PV','Batterie')
    set(gca,'xticklabel',mois);
    hold on

figure(2)
temps = [0.25:0.25:24];
subplot(2,3,1)
plot(temps,sol{4}.s_tj(1,:))
title('�tat d''activation pour les f�netres pour 1 journ�e')
xlabel('Heure')
ylabel('�tat d''activation')

subplot(2,3,2)
plot(temps,sol{4}.s_tj(2,:))
title('�tat d''activation pour la pompe pour 1 journ�e')
xlabel('Heure')
ylabel('�tat d''activation')

subplot(2,3,3)
plot(temps,sol{4}.s_tj(3,:))
title('�tat d''activation pour le d�shumidificateur pour 1 journ�e')
xlabel('Heure')
ylabel('�tat d''activation')

subplot(2,3,4)
plot(temps,sol{4}.s_tj(4,:))
title('�tat d''activation pour le chauffage pour 1 journ�e')
xlabel('Heure')
ylabel('�tat d''activation')

subplot(2,3,5)
plot(temps,sol{4}.s_tj(5,:))
title('�tat d''activation pour la climitisation pour 1 journ�e')
xlabel('Heure')
ylabel('�tat d''activation')

