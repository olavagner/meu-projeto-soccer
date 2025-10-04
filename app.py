import streamlit as st
import requests
from bs4 import BeautifulSoup
import pandas as pd
import concurrent.futures
from datetime import datetime, timedelta
import re
import numpy as np
from scipy.stats import poisson
import warnings

warnings.filterwarnings('ignore')


# 🔥 CLASSE ANALISADOR PICO MÁXIMO (INTEGRADA)
class AnalisadorPicoMaximo:
    def __init__(self, dados_historicos):
        self.dados = dados_historicos
        self.pesos_progressivos = [0.08, 0.12, 0.16, 0.20, 0.25, 0.35, 0.50, 0.65, 0.80, 0.95]

    def calcular_estatisticas_avancadas(self, equipe, num_jogos=15):
        colunas_necessarias = ['Casa', 'Fora', 'HT', 'FT']
        colunas_existentes = [col for col in colunas_necessarias if col in self.dados.columns]

        if len(colunas_existentes) < 4:
            return None

        jogos_equipe = self.dados[
            (self.dados['Casa'] == equipe) | (self.dados['Fora'] == equipe)
            ].tail(num_jogos)

        if len(jogos_equipe) == 0:
            return None

        estatisticas = {
            'gols_feitos_ht': [], 'gols_sofridos_ht': [], 'gols_feitos_ft': [], 'gols_sofridos_ft': [],
            'over_05_ht': [], 'over_15_ht': [], 'over_05_ft': [], 'over_15_ft': [],
            'over_25_ft': [], 'over_35_ft': [], 'btts': [], 'goals_1_5_plus': [], 'goals_2_5_plus': [],
        }

        for i, (idx, jogo) in enumerate(jogos_equipe.iterrows()):
            peso = self.pesos_progressivos[i] if i < len(self.pesos_progressivos) else 0.1
            is_casa = jogo['Casa'] == equipe

            # Extrair gols do HT e FT
            gols_casa_ht, gols_fora_ht = self.extrair_gols_ht(jogo['HT'])
            gols_casa_ft, gols_fora_ft = self.extrair_gols_ft(jogo['FT'])

            if is_casa:
                gols_feitos_ht = gols_casa_ht
                gols_sofridos_ht = gols_fora_ht
                gols_feitos_ft = gols_casa_ft
                gols_sofridos_ft = gols_fora_ft
            else:
                gols_feitos_ht = gols_fora_ht
                gols_sofridos_ht = gols_casa_ht
                gols_feitos_ft = gols_fora_ft
                gols_sofridos_ft = gols_casa_ft

            try:
                gols_feitos_ht = float(gols_feitos_ht) if pd.notna(gols_feitos_ht) else 0
                gols_sofridos_ht = float(gols_sofridos_ht) if pd.notna(gols_sofridos_ht) else 0
                gols_feitos_ft = float(gols_feitos_ft) if pd.notna(gols_feitos_ft) else 0
                gols_sofridos_ft = float(gols_sofridos_ft) if pd.notna(gols_sofridos_ft) else 0
            except:
                gols_feitos_ht, gols_sofridos_ht, gols_feitos_ft, gols_sofridos_ft = 0, 0, 0, 0

            estatisticas['gols_feitos_ht'].append(gols_feitos_ht * peso)
            estatisticas['gols_sofridos_ht'].append(gols_sofridos_ht * peso)
            estatisticas['gols_feitos_ft'].append(gols_feitos_ft * peso)
            estatisticas['gols_sofridos_ft'].append(gols_sofridos_ft * peso)

            total_ht = gols_feitos_ht + gols_sofridos_ht
            total_ft = gols_feitos_ft + gols_sofridos_ft

            estatisticas['over_05_ht'].append(1 if total_ht > 0.5 else 0)
            estatisticas['over_15_ht'].append(1 if total_ht > 1.5 else 0)
            estatisticas['over_05_ft'].append(1 if total_ft > 0.5 else 0)
            estatisticas['over_15_ft'].append(1 if total_ft > 1.5 else 0)
            estatisticas['over_25_ft'].append(1 if total_ft > 2.5 else 0)
            estatisticas['over_35_ft'].append(1 if total_ft > 3.5 else 0)
            estatisticas['btts'].append(1 if gols_feitos_ft > 0 and gols_sofridos_ft > 0 else 0)
            estatisticas['goals_1_5_plus'].append(1 if gols_feitos_ft >= 1.5 else 0)
            estatisticas['goals_2_5_plus'].append(1 if gols_feitos_ft >= 2.5 else 0)

        resultados = {}
        for key, valores in estatisticas.items():
            if valores:
                resultados[key] = sum(valores) / sum(self.pesos_progressivos[:len(valores)])
            else:
                resultados[key] = 0

        return resultados

    def extrair_gols_ht(self, ht_value):
        """Extrai gols do HT (formato: '1-0' ou '1-0 (0-0)')"""
        try:
            if pd.isna(ht_value) or ht_value == '':
                return 0, 0

            # Remove conteúdo entre parênteses se existir
            ht_limpo = re.sub(r'\([^)]*\)', '', ht_value).strip()

            if '-' in ht_limpo:
                partes = ht_limpo.split('-')
                if len(partes) == 2:
                    return float(partes[0].strip()), float(partes[1].strip())
            return 0, 0
        except:
            return 0, 0

    def extrair_gols_ft(self, ft_value):
        """Extrai gols do FT (formato: '1-0')"""
        try:
            if pd.isna(ft_value) or ft_value == '':
                return 0, 0

            if '-' in ft_value:
                partes = ft_value.split('-')
                if len(partes) == 2:
                    return float(partes[0].strip()), float(partes[1].strip())
            return 0, 0
        except:
            return 0, 0

    def simular_jogo_monte_carlo(self, stats_casa, stats_fora, num_simulacoes=50000):
        if not stats_casa or not stats_fora:
            return None

        lambda_casa_ft = max(0.1, (stats_casa['gols_feitos_ft'] * 0.6 + stats_fora['gols_sofridos_ft'] * 0.4))
        lambda_fora_ft = max(0.1, (stats_fora['gols_feitos_ft'] * 0.6 + stats_casa['gols_sofridos_ft'] * 0.4))

        fator_casa = 1.15
        lambda_casa_ft *= fator_casa
        lambda_fora_ft *= 0.85

        try:
            gols_casa_ft = poisson.rvs(mu=lambda_casa_ft, size=num_simulacoes)
            gols_fora_ft = poisson.rvs(mu=lambda_fora_ft, size=num_simulacoes)

            gols_ht_casa = np.random.binomial(gols_casa_ft, 0.4, size=num_simulacoes)
            gols_ht_fora = np.random.binomial(gols_fora_ft, 0.4, size=num_simulacoes)

            resultados = {
                'gols_ht_casa': gols_ht_casa, 'gols_ht_fora': gols_ht_fora,
                'gols_ft_casa': gols_casa_ft, 'gols_ft_fora': gols_fora_ft,
                'total_ht': gols_ht_casa + gols_ht_fora, 'total_ft': gols_casa_ft + gols_fora_ft
            }

            return resultados
        except:
            return None

    def calcular_probabilidades_pico_maximo(self, casa, fora):
        stats_casa = self.calcular_estatisticas_avancadas(casa)
        stats_fora = self.calcular_estatisticas_avancadas(fora)

        if not stats_casa or not stats_fora:
            return None

        simulacao = self.simular_jogo_monte_carlo(stats_casa, stats_fora, 100000)

        if not simulacao:
            return None

        prob = {}

        try:
            # MERCADOS HT
            prob['Gols Esperados HT'] = np.mean(simulacao['total_ht'])
            prob['Over 0.5 HT'] = max(60, min(95, np.mean(simulacao['total_ht'] > 0.5) * 100 * 1.08))
            prob['Over 1.5 HT'] = max(40, min(85, np.mean(simulacao['total_ht'] > 1.5) * 100 * 1.06))
            prob['Casa Marca HT'] = max(50, min(90, np.mean(simulacao['gols_ht_casa'] > 0) * 100 * 1.05))
            prob['Fora Marca HT'] = max(45, min(85, np.mean(simulacao['gols_ht_fora'] > 0) * 100 * 1.05))

            # MERCADOS FT
            prob['Gols Esperados FT'] = np.mean(simulacao['total_ft'])
            prob['Over 0.5 FT'] = max(85, min(99, np.mean(simulacao['total_ft'] > 0.5) * 100 * 1.02))
            prob['Over 1.5 FT'] = max(70, min(95, np.mean(simulacao['total_ft'] > 1.5) * 100 * 1.04))
            prob['Over 2.5 FT'] = max(50, min(90, np.mean(simulacao['total_ft'] > 2.5) * 100 * 1.05))
            prob['Over 3.5 FT'] = max(25, min(75, np.mean(simulacao['total_ft'] > 3.5) * 100 * 1.06))
            prob['Over 4.5 FT'] = max(10, min(50, np.mean(simulacao['total_ft'] > 4.5) * 100 * 1.08))

            # BTTS
            btts_prob = np.mean((simulacao['gols_ft_casa'] > 0) & (simulacao['gols_ft_fora'] > 0)) * 100
            prob['BTTS FT'] = max(40, min(85, btts_prob * 1.07))

            # BTTS & Over 2.5
            btts_over25_prob = np.mean(
                (simulacao['gols_ft_casa'] > 0) &
                (simulacao['gols_ft_fora'] > 0) &
                (simulacao['total_ft'] > 2.5)
            ) * 100
            prob['BTTS & Over 2.5'] = max(25, min(70, btts_over25_prob * 1.08))

            # Equipe marca 1.5+
            prob['Casa Marca 1.5'] = max(30, min(80, np.mean(simulacao['gols_ft_casa'] >= 1.5) * 100 * 1.09))
            prob['Fora Marca 1.5'] = max(25, min(70, np.mean(simulacao['gols_ft_fora'] >= 1.5) * 100 * 1.09))

            # Probabilidades básicas para vitória/empate
            prob['Casa Vence'] = np.mean(simulacao['gols_ft_casa'] > simulacao['gols_ft_fora']) * 100
            prob['Empate'] = np.mean(simulacao['gols_ft_casa'] == simulacao['gols_ft_fora']) * 100
            prob['Fora Vence'] = np.mean(simulacao['gols_ft_casa'] < simulacao['gols_ft_fora']) * 100

            return prob
        except Exception as e:
            return None


# 🔥 CLASSE PARA DICAS ESTATÍSTICAS - ATUALIZADA
class AnalisadorDicasEstatisticas:
    def __init__(self, dados_historicos):
        self.dados = dados_historicos
        self.mercados_config = {
            'Over 0.5 HT': {'nome': 'Over 0.5 HT', 'icone': '⚡', 'limite': 75, 'tipo': 'over_ht', 'linha': 0.5},
            'Over 1.5 HT': {'nome': 'Over 1.5 HT', 'icone': '⚡', 'limite': 45, 'tipo': 'over_ht', 'linha': 1.5},
            'Over 0.5 FT': {'nome': 'Over 0.5 FT', 'icone': '🎯', 'limite': 85, 'tipo': 'over_ft', 'linha': 0.5},
            'Over 1.5 FT': {'nome': 'Over 1.5 FT', 'icone': '🎯', 'limite': 70, 'tipo': 'over_ft', 'linha': 1.5},
            'Over 2.5 FT': {'nome': 'Over 2.5 FT', 'icone': '🎯', 'limite': 55, 'tipo': 'over_ft', 'linha': 2.5},
            'Over 3.5 FT': {'nome': 'Over 3.5 FT', 'icone': '🎯', 'limite': 35, 'tipo': 'over_ft', 'linha': 3.5},
            'BTTS FT': {'nome': 'BTTS FT', 'icone': '🔀', 'limite': 60, 'tipo': 'btts', 'linha': None},
            'BTTS & Over 2.5': {'nome': 'BTTS & Over 2.5', 'icone': '🔥', 'limite': 45, 'tipo': 'combinado',
                                'linha': None},
            'Casa Marca 1.5': {'nome': 'Casa Marca 1.5+', 'icone': '🏠', 'limite': 50, 'tipo': 'equipe_ataque',
                               'linha': 1.5},
            'Fora Marca 1.5': {'nome': 'Fora Marca 1.5+', 'icone': '✈️', 'limite': 40, 'tipo': 'equipe_ataque',
                               'linha': 1.5},
            'Casa Vence': {'nome': 'Vitória Casa', 'icone': '🏠', 'limite': 65, 'tipo': 'resultado', 'linha': None},
            'Fora Vence': {'nome': 'Vitória Fora', 'icone': '✈️', 'limite': 55, 'tipo': 'resultado', 'linha': None}
        }

    def gerar_dicas_jogo(self, casa, fora, mercado_filtro=None):
        """Gera dicas estatísticas para um jogo específico com filtro por mercado"""
        stats_casa = self.calcular_estatisticas_equipe(casa)
        stats_fora = self.calcular_estatisticas_equipe(fora)

        if not stats_casa or not stats_fora:
            return []

        dicas = []

        # Filtrar mercados se especificado
        mercados_para_analisar = self.mercados_config
        if mercado_filtro and mercado_filtro != "Todos":
            mercados_para_analisar = {k: v for k, v in self.mercados_config.items()
                                      if v['nome'] == mercado_filtro}

        # Calcular probabilidades combinadas
        for mercado, config in mercados_para_analisar.items():
            prob_casa = stats_casa.get(mercado, 0)
            prob_fora = stats_fora.get(mercado, 0)

            # Média ponderada considerando força das equipes
            probabilidade_combinada = (prob_casa * 0.6 + prob_fora * 0.4)

            if probabilidade_combinada >= config['limite']:
                dicas.append({
                    'mercado': config['nome'],
                    'icone': config['icone'],
                    'probabilidade': probabilidade_combinada,
                    'casa_percent': prob_casa,
                    'fora_percent': prob_fora,
                    'tipo': config['tipo'],
                    'linha': config['linha']
                })

        # 🔥 SELEÇÃO INTELIGENTE - Evitar mercados redundantes
        dicas_filtradas = self._filtrar_mercados_redundantes(dicas)

        # Ordenar por probabilidade (maior primeiro)
        dicas_filtradas.sort(key=lambda x: x['probabilidade'], reverse=True)
        return dicas_filtradas

    def _filtrar_mercados_redundantes(self, dicas):
        """Filtra mercados redundantes, mantendo apenas a linha mais alta"""
        if not dicas:
            return []

        # Agrupar por tipo de mercado
        mercados_por_tipo = {}
        for dica in dicas:
            tipo = dica['tipo']
            if tipo not in mercados_por_tipo:
                mercados_por_tipo[tipo] = []
            mercados_por_tipo[tipo].append(dica)

        dicas_finais = []

        # Para cada tipo, manter apenas o mercado com linha mais alta
        for tipo, mercados in mercados_por_tipo.items():
            if tipo in ['over_ht', 'over_ft']:
                # Para mercados Over, manter apenas o com linha mais alta
                mercado_maior_linha = max(mercados, key=lambda x: x['linha'] if x['linha'] else 0)
                dicas_finais.append(mercado_maior_linha)
            else:
                # Para outros tipos, manter todos
                dicas_finais.extend(mercados)

        return dicas_finais

    def calcular_estatisticas_equipe(self, equipe, num_jogos=10):
        """Calcula estatísticas recentes de uma equipe"""
        jogos_equipe = self.dados[
            (self.dados['Casa'] == equipe) | (self.dados['Fora'] == equipe)
            ].tail(num_jogos)

        if len(jogos_equipe) == 0:
            return None

        stats = {
            'Casa Vence': 0, 'Fora Vence': 0,
            'Over 0.5 HT': 0, 'Over 1.5 HT': 0,
            'Over 0.5 FT': 0, 'Over 1.5 FT': 0, 'Over 2.5 FT': 0, 'Over 3.5 FT': 0,
            'BTTS FT': 0, 'BTTS & Over 2.5': 0,
            'Casa Marca 1.5': 0, 'Fora Marca 1.5': 0,
            'total_jogos': len(jogos_equipe)
        }

        for idx, jogo in jogos_equipe.iterrows():
            is_casa = jogo['Casa'] == equipe

            # Extrair gols
            gols_casa_ht, gols_fora_ht = self.extrair_gols_ht(jogo['HT'])
            gols_casa_ft, gols_fora_ft = self.extrair_gols_ft(jogo['FT'])

            if is_casa:
                gols_feitos_ft = gols_casa_ft
                gols_sofridos_ft = gols_fora_ft
            else:
                gols_feitos_ft = gols_fora_ft
                gols_sofridos_ft = gols_casa_ft

            # Calcular estatísticas
            total_ht = gols_casa_ht + gols_fora_ht
            total_ft = gols_casa_ft + gols_fora_ft

            # Vitórias
            if is_casa and gols_casa_ft > gols_fora_ft:
                stats['Casa Vence'] += 1
            elif not is_casa and gols_fora_ft > gols_casa_ft:
                stats['Fora Vence'] += 1

            # Mercados HT
            if total_ht > 0.5: stats['Over 0.5 HT'] += 1
            if total_ht > 1.5: stats['Over 1.5 HT'] += 1

            # Mercados FT
            if total_ft > 0.5: stats['Over 0.5 FT'] += 1
            if total_ft > 1.5: stats['Over 1.5 FT'] += 1
            if total_ft > 2.5: stats['Over 2.5 FT'] += 1
            if total_ft > 3.5: stats['Over 3.5 FT'] += 1
            if gols_casa_ft > 0 and gols_fora_ft > 0: stats['BTTS FT'] += 1
            if (gols_casa_ft > 0 and gols_fora_ft > 0) and total_ft > 2.5: stats['BTTS & Over 2.5'] += 1
            if gols_feitos_ft >= 1.5:
                if is_casa:
                    stats['Casa Marca 1.5'] += 1
                else:
                    stats['Fora Marca 1.5'] += 1

        # Converter para percentuais
        for key in stats:
            if key != 'total_jogos' and stats['total_jogos'] > 0:
                stats[key] = (stats[key] / stats['total_jogos']) * 100

        return stats

    def extrair_gols_ht(self, ht_value):
        """Extrai gols do HT"""
        try:
            if pd.isna(ht_value) or ht_value == '':
                return 0, 0
            ht_limpo = re.sub(r'\([^)]*\)', '', ht_value).strip()
            if '-' in ht_limpo:
                partes = ht_limpo.split('-')
                if len(partes) == 2:
                    return float(partes[0].strip()), float(partes[1].strip())
            return 0, 0
        except:
            return 0, 0

    def extrair_gols_ft(self, ft_value):
        """Extrai gols do FT"""
        try:
            if pd.isna(ft_value) or ft_value == '':
                return 0, 0
            if '-' in ft_value:
                partes = ft_value.split('-')
                if len(partes) == 2:
                    return float(partes[0].strip()), float(partes[1].strip())
            return 0, 0
        except:
            return 0, 0


# 🔥 CLASSE PARA ALERTAS INTELIGENTES - COMPLETA E CORRIGIDA
class AnalisadorAlertasInteligentes:
    def __init__(self, dados_historicos):
        self.dados = dados_historicos
        self.mercados = {
            'Vitorias': {'nome': 'Vitórias', 'icone': '✅', 'tipo': 'vitoria'},
            'Derrotas': {'nome': 'Derrotas', 'icone': '❌', 'tipo': 'derrota'},
            'Over 0.5 HT': {'nome': 'Over 0.5 HT', 'icone': '⚡', 'tipo': 'over_ht', 'linha': 0.5},
            'Over 1.5 HT': {'nome': 'Over 1.5 HT', 'icone': '⚡', 'tipo': 'over_ht', 'linha': 1.5},
            'BTTS HT': {'nome': 'BTTS HT', 'icone': '🔀', 'tipo': 'btts_ht'},
            'Over 1.5 FT': {'nome': 'Over 1.5 FT', 'icone': '🎯', 'tipo': 'over_ft', 'linha': 1.5},
            'Over 2.5 FT': {'nome': 'Over 2.5 FT', 'icone': '🎯', 'tipo': 'over_ft', 'linha': 2.5},
            'Over 3.5 FT': {'nome': 'Over 3.5 FT', 'icone': '🎯', 'tipo': 'over_ft', 'linha': 3.5},
            'BTTS FT': {'nome': 'BTTS FT', 'icone': '🔀', 'tipo': 'btts_ft'},
            'Casa Marca 1.5': {'nome': 'Casa Marca 1.5+', 'icone': '🏠', 'tipo': 'equipe_ataque', 'linha': 1.5},
            'Fora Marca 1.5': {'nome': 'Fora Marca 1.5+', 'icone': '✈️', 'tipo': 'equipe_ataque', 'linha': 1.5}
        }

    def extrair_gols_ht(self, ht_value):
        """Extrai gols do HT - PEGAR DADOS DENTRO DOS PARÊNTESES"""
        try:
            if pd.isna(ht_value) or ht_value == '' or ht_value == '-':
                return 0, 0

            # Converter para string
            ht_str = str(ht_value).strip()

            # BUSCAR DADOS DENTRO DOS PARÊNTESES - gols reais do HT
            padrao_parenteses = r'\((\d+)-(\d+)\)'
            match = re.search(padrao_parenteses, ht_str)

            if match:
                # Encontrou dados entre parênteses - usar esses (são os gols reais do HT)
                gols_casa = float(match.group(1))
                gols_fora = float(match.group(2))
                return gols_casa, gols_fora
            else:
                # Se não tem parênteses, tentar extrair do formato básico
                if '-' in ht_str:
                    partes = ht_str.split('-')
                    if len(partes) == 2:
                        gols_casa = re.sub(r'[^\d]', '', partes[0].strip())
                        gols_fora = re.sub(r'[^\d]', '', partes[1].strip())

                        gols_casa = float(gols_casa) if gols_casa.isdigit() else 0
                        gols_fora = float(gols_fora) if gols_fora.isdigit() else 0

                        return gols_casa, gols_fora

            return 0, 0

        except Exception as e:
            return 0, 0

    def extrair_gols_ft(self, ft_value):
        """Extrai gols do FT - PEGAR DADOS FORA DOS PARÊNTESES"""
        try:
            if pd.isna(ft_value) or ft_value == '' or ft_value == '-':
                return 0, 0

            # Converter para string
            ft_str = str(ft_value).strip()

            # Para FT, usar os dados principais (fora dos parênteses)
            # Remover conteúdo entre parênteses se existir
            ft_limpo = re.sub(r'\([^)]*\)', '', ft_str).strip()

            if '-' in ft_limpo:
                partes = ft_limpo.split('-')
                if len(partes) == 2:
                    gols_casa = re.sub(r'[^\d]', '', partes[0].strip())
                    gols_fora = re.sub(r'[^\d]', '', partes[1].strip())

                    gols_casa = float(gols_casa) if gols_casa.isdigit() else 0
                    gols_fora = float(gols_fora) if gols_fora.isdigit() else 0

                    return gols_casa, gols_fora

            return 0, 0

        except Exception as e:
            return 0, 0

    def verificar_mercado_jogo_liga(self, jogo, mercado):
        """Verifica se o mercado foi atendido no jogo para análise da liga"""
        try:
            # Extrair gols
            gols_casa_ht, gols_fora_ht = self.extrair_gols_ht(jogo['HT'])
            gols_casa_ft, gols_fora_ft = self.extrair_gols_ft(jogo['FT'])

            # Verificar mercado
            if mercado == 'Vitorias':
                return gols_casa_ft > gols_fora_ft

            elif mercado == 'Derrotas':
                return gols_casa_ft < gols_fora_ft

            elif mercado == 'Over 0.5 HT':
                total_ht = gols_casa_ht + gols_fora_ht
                return total_ht > 0.5

            elif mercado == 'Over 1.5 HT':
                total_ht = gols_casa_ht + gols_fora_ht
                return total_ht > 1.5

            elif mercado == 'BTTS HT':
                return gols_casa_ht > 0 and gols_fora_ht > 0

            elif mercado == 'Over 1.5 FT':
                total_ft = gols_casa_ft + gols_fora_ft
                return total_ft > 1.5

            elif mercado == 'Over 2.5 FT':
                total_ft = gols_casa_ft + gols_fora_ft
                return total_ft > 2.5

            elif mercado == 'Over 3.5 FT':
                total_ft = gols_casa_ft + gols_fora_ft
                return total_ft > 3.5

            elif mercado == 'BTTS FT':
                return gols_casa_ft > 0 and gols_fora_ft > 0

            elif mercado == 'Casa Marca 1.5':
                return gols_casa_ft >= 1.5

            elif mercado == 'Fora Marca 1.5':
                return gols_fora_ft >= 1.5

            return False

        except Exception as e:
            return False

    def verificar_mercado_jogo_equipe(self, jogo, equipe, mercado):
        """Verifica se o mercado foi atendido no jogo para a equipe específica"""
        try:
            is_casa = jogo['Casa'] == equipe

            # Extrair gols
            gols_casa_ht, gols_fora_ht = self.extrair_gols_ht(jogo['HT'])
            gols_casa_ft, gols_fora_ft = self.extrair_gols_ft(jogo['FT'])

            if is_casa:
                gols_feitos_ft = gols_casa_ft
                gols_sofridos_ft = gols_fora_ft
                gols_feitos_ht = gols_casa_ht
                gols_sofridos_ht = gols_fora_ht
            else:
                gols_feitos_ft = gols_fora_ft
                gols_sofridos_ft = gols_casa_ft
                gols_feitos_ht = gols_fora_ht
                gols_sofridos_ht = gols_casa_ht

            # Verificar mercado
            if mercado == 'Vitorias':
                if is_casa:
                    return gols_casa_ft > gols_fora_ft
                else:
                    return gols_fora_ft > gols_casa_ft

            elif mercado == 'Derrotas':
                if is_casa:
                    return gols_casa_ft < gols_fora_ft
                else:
                    return gols_fora_ft < gols_casa_ft

            elif mercado == 'Over 0.5 HT':
                total_ht = gols_casa_ht + gols_fora_ht
                return total_ht > 0.5

            elif mercado == 'Over 1.5 HT':
                total_ht = gols_casa_ht + gols_fora_ht
                return total_ht > 1.5

            elif mercado == 'BTTS HT':
                return gols_casa_ht > 0 and gols_fora_ht > 0

            elif mercado == 'Over 1.5 FT':
                total_ft = gols_casa_ft + gols_fora_ft
                return total_ft > 1.5

            elif mercado == 'Over 2.5 FT':
                total_ft = gols_casa_ft + gols_fora_ft
                return total_ft > 2.5

            elif mercado == 'Over 3.5 FT':
                total_ft = gols_casa_ft + gols_fora_ft
                return total_ft > 3.5

            elif mercado == 'BTTS FT':
                return gols_casa_ft > 0 and gols_fora_ft > 0

            elif mercado == 'Casa Marca 1.5':
                if is_casa:
                    return gols_feitos_ft >= 1.5
                else:
                    return gols_sofridos_ft >= 1.5

            elif mercado == 'Fora Marca 1.5':
                if not is_casa:
                    return gols_feitos_ft >= 1.5
                else:
                    return gols_sofridos_ft >= 1.5

            return None

        except Exception as e:
            return None

    def calcular_estatisticas_equipe_geral(self, equipe, mercado):
        """Calcula estatísticas gerais de uma equipe para um mercado específico"""
        jogos_equipe = self.dados[
            (self.dados['Casa'] == equipe) | (self.dados['Fora'] == equipe)
            ].copy()

        if len(jogos_equipe) == 0:
            return 0, 0, []

        acertos = 0
        total_jogos_validos = 0
        ultimos_resultados = []

        for idx, jogo in jogos_equipe.iterrows():
            # Para mercados HT, verificar se tem dados válidos
            if mercado in ['Over 0.5 HT', 'Over 1.5 HT', 'BTTS HT']:
                gols_casa_ht, gols_fora_ht = self.extrair_gols_ht(jogo['HT'])
                if gols_casa_ht == 0 and gols_fora_ht == 0:
                    continue  # Pular jogo sem dados HT válidos

            bateu_mercado = self.verificar_mercado_jogo_equipe(jogo, equipe, mercado)
            if bateu_mercado is not None:
                total_jogos_validos += 1
                if bateu_mercado:
                    acertos += 1
                # Guardar últimos 5 resultados (mais recentes primeiro)
                if len(ultimos_resultados) < 5:
                    ultimos_resultados.append('🟢' if bateu_mercado else '🔴')

        # Reverter para ter os mais recentes primeiro
        ultimos_resultados = ultimos_resultados[::-1]

        # Completar com ⚫ se não tiver 5 jogos
        while len(ultimos_resultados) < 5:
            ultimos_resultados.append('⚫')

        taxa_acerto = (acertos / total_jogos_validos * 100) if total_jogos_validos > 0 else 0

        return taxa_acerto, total_jogos_validos, ultimos_resultados

    def calcular_estatisticas_liga_geral(self, liga, mercado):
        """Calcula estatísticas gerais de uma liga para um mercado específico"""
        dados_liga = self.dados[self.dados['Competição'] == liga].copy()

        if dados_liga.empty:
            return 0, 0

        total_jogos = 0
        acertos = 0
        jogos_com_dados_ht = 0

        for idx, jogo in dados_liga.iterrows():
            # Verificar se tem dados HT válidos para mercados HT
            if mercado in ['Over 0.5 HT', 'Over 1.5 HT', 'BTTS HT']:
                gols_casa_ht, gols_fora_ht = self.extrair_gols_ht(jogo['HT'])
                if gols_casa_ht == 0 and gols_fora_ht == 0:
                    continue  # Pular jogo sem dados HT válidos
                jogos_com_dados_ht += 1

            total_jogos += 1
            if self.verificar_mercado_jogo_liga(jogo, mercado):
                acertos += 1

        # Para mercados HT, usar apenas jogos com dados válidos
        if mercado in ['Over 0.5 HT', 'Over 1.5 HT', 'BTTS HT']:
            total_jogos = jogos_com_dados_ht

        taxa_acerto = (acertos / total_jogos * 100) if total_jogos > 0 else 0

        return taxa_acerto, total_jogos

    def gerar_ranking_mercado(self, mercado, competicao=None):
        """Gera ranking completo para um mercado específico"""
        todas_equipes = list(set(self.dados['Casa'].unique()) | set(self.dados['Fora'].unique()))
        todas_ligas = self.dados['Competição'].unique()

        # Calcular estatísticas das ligas
        ranking_ligas = []
        for liga in todas_ligas:
            taxa_liga, total_jogos_liga = self.calcular_estatisticas_liga_geral(liga, mercado)

            # Para mercados HT, exigir pelo menos 3 jogos com dados HT válidos
            min_jogos = 5
            if mercado in ['Over 0.5 HT', 'Over 1.5 HT', 'BTTS HT']:
                min_jogos = 3

            if total_jogos_liga >= min_jogos:
                ranking_ligas.append({
                    'Liga': liga,
                    'Taxa': taxa_liga,
                    'Jogos': total_jogos_liga,
                    '_taxa_num': taxa_liga
                })

        # Ordenar ligas por taxa
        ranking_ligas.sort(key=lambda x: x['_taxa_num'], reverse=True)

        # Calcular estatísticas das equipes
        resultados_equipes = []
        for equipe in todas_equipes:
            taxa, total_jogos, ultimos_5 = self.calcular_estatisticas_equipe_geral(equipe, mercado)

            # MÍNIMO DE JOGOS ajustado para mercados HT
            min_jogos_equipe = 5
            if mercado in ['Over 0.5 HT', 'Over 1.5 HT', 'BTTS HT']:
                min_jogos_equipe = 3

            if total_jogos >= min_jogos_equipe:
                # Encontrar liga da equipe (mais frequente)
                jogos_equipe = self.dados[
                    (self.dados['Casa'] == equipe) | (self.dados['Fora'] == equipe)
                    ]
                if not jogos_equipe.empty:
                    liga = jogos_equipe['Competição'].mode()[0]

                    # Se filtro por competição, filtrar equipes
                    if competicao and competicao != "Todas":
                        if liga != competicao:
                            continue

                    resultados_equipes.append({
                        'Equipe': equipe,
                        'Liga': liga,
                        'Jogos': total_jogos,
                        'Acertos': int((taxa / 100) * total_jogos),
                        'Taxa': taxa,
                        'Últimos 5': ' '.join(ultimos_5),
                        '_taxa_num': taxa
                    })

        # Ordenar equipes por taxa
        resultados_equipes.sort(key=lambda x: x['_taxa_num'], reverse=True)

        return resultados_equipes, ranking_ligas

    def extrair_gols_ht(self, ht_value):
        """Extrai gols do HT - CORRIGIDO"""
        try:
            if pd.isna(ht_value) or ht_value == '' or ht_value == '-':
                return 0, 0

            # Limpar string - remover parênteses e conteúdo dentro
            ht_limpo = re.sub(r'\([^)]*\)', '', str(ht_value)).strip()

            if '-' in ht_limpo:
                partes = ht_limpo.split('-')
                if len(partes) == 2:
                    gols_casa = float(partes[0].strip()) if partes[0].strip().isdigit() else 0
                    gols_fora = float(partes[1].strip()) if partes[1].strip().isdigit() else 0
                    return gols_casa, gols_fora

            return 0, 0

        except Exception as e:
            return 0, 0

    def extrair_gols_ft(self, ft_value):
        """Extrai gols do FT - CORRIGIDO"""
        try:
            if pd.isna(ft_value) or ft_value == '' or ft_value == '-':
                return 0, 0

            if '-' in str(ft_value):
                partes = str(ft_value).split('-')
                if len(partes) == 2:
                    gols_casa = float(partes[0].strip()) if partes[0].strip().isdigit() else 0
                    gols_fora = float(partes[1].strip()) if partes[1].strip().isdigit() else 0
                    return gols_casa, gols_fora

            return 0, 0

        except Exception as e:
            return 0, 0


# 🔥 FUNÇÃO PARA ADICIONAR ANÁLISE PICO MÁXIMO AOS JOGOS
def adicionar_analise_pico_maximo(df_jogos, base_historica):
    if df_jogos.empty or base_historica.empty:
        return df_jogos

    analisador = AnalisadorPicoMaximo(base_historica)
    novas_colunas = []

    if len(df_jogos) > 0:
        progress_bar = st.progress(0)
        total_jogos = len(df_jogos)

        for idx, jogo in df_jogos.iterrows():
            progresso = min((idx + 1) / total_jogos, 1.0)
            progress_bar.progress(progresso)

            try:
                probabilidades = analisador.calcular_probabilidades_pico_maximo(jogo['Casa'], jogo['Fora'])

                if probabilidades:
                    resultado_jogo = {
                        'Casa Vence': f"{probabilidades['Casa Vence']:.1f}%",
                        'Empate': f"{probabilidades['Empate']:.1f}%",
                        'Fora Vence': f"{probabilidades['Fora Vence']:.1f}%",
                        'Gols HT': f"{probabilidades['Gols Esperados HT']:.2f}",
                        'Over 0.5 HT': f"{probabilidades['Over 0.5 HT']:.1f}%",
                        'Over 1.5 HT': f"{probabilidades['Over 1.5 HT']:.1f}%",
                        'Casa Marca HT': f"{probabilidades['Casa Marca HT']:.1f}%",
                        'Fora Marca HT': f"{probabilidades['Fora Marca HT']:.1f}%",
                        'Gols FT': f"{probabilidades['Gols Esperados FT']:.2f}",
                        'Over 0.5 FT': f"{probabilidades['Over 0.5 FT']:.1f}%",
                        'Over 1.5 FT': f"{probabilidades['Over 1.5 FT']:.1f}%",
                        'Over 2.5 FT': f"{probabilidades['Over 2.5 FT']:.1f}%",
                        'Over 3.5 FT': f"{probabilidades['Over 3.5 FT']:.1f}%",
                        'Over 4.5 FT': f"{probabilidades['Over 4.5 FT']:.1f}%",
                        'Casa Marca 1.5': f"{probabilidades['Casa Marca 1.5']:.1f}%",
                        'Fora Marca 1.5': f"{probabilidades['Fora Marca 1.5']:.1f}%",
                        'Btts FT': f"{probabilidades['BTTS FT']:.1f}%",
                        'Btts & Over 2.5': f"{probabilidades['BTTS & Over 2.5']:.1f}%"
                    }
                    novas_colunas.append(resultado_jogo)
                else:
                    valores_padrao = {
                        'Casa Vence': "-", 'Empate': "-", 'Fora Vence': "-",
                        'Gols HT': "-", 'Over 0.5 HT': "-", 'Over 1.5 HT': "-",
                        'Casa Marca HT': "-", 'Fora Marca HT': "-", 'Gols FT': "-",
                        'Over 0.5 FT': "-", 'Over 1.5 FT': "-", 'Over 2.5 FT': "-",
                        'Over 3.5 FT': "-", 'Over 4.5 FT': "-", 'Casa Marca 1.5': "-",
                        'Fora Marca 1.5': "-", 'Btts FT': "-", 'Btts & Over 2.5': "-"
                    }
                    novas_colunas.append(valores_padrao)

            except Exception as e:
                valores_erro = {f"Erro": "Erro" for _ in range(18)}
                novas_colunas.append(valores_erro)

        progress_bar.empty()

    if novas_colunas:
        df_com_analise = df_jogos.copy()
        for coluna in novas_colunas[0].keys():
            df_com_analise[coluna] = [jogo[coluna] for jogo in novas_colunas]

        return df_com_analise

    return df_jogos


# 🔥 FUNÇÕES ORIGINAIS DO SEU CÓDIGO (MANTIDAS)
def traduzir_data(data_ingles):
    dias_semana = {
        'Mon': 'Seg', 'Tue': 'Ter', 'Wed': 'Qua', 'Thu': 'Qui',
        'Fri': 'Sex', 'Sat': 'Sáb', 'Sun': 'Dom'
    }
    meses = {
        'Jan': 'Jan', 'Feb': 'Fev', 'Mar': 'Mar', 'Apr': 'Abr',
        'May': 'Mai', 'Jun': 'Jun', 'Jul': 'Jul', 'Aug': 'Ago',
        'Sep': 'Set', 'Oct': 'Out', 'Nov': 'Nov', 'Dec': 'Dez'
    }
    try:
        data_ingles = data_ingles.replace('Percentages', '').strip()
        partes = data_ingles.split()
        if len(partes) == 3:
            dia_semana_eng = partes[0]
            dia_mes = partes[1]
            mes_eng = partes[2]
            dia_semana_pt = dias_semana.get(dia_semana_eng, dia_semana_eng)
            mes_pt = meses.get(mes_eng, mes_eng)
            return f"{dia_semana_pt} {dia_mes} {mes_pt}"
        else:
            return data_ingles
    except:
        return data_ingles


def limpar_ht(ht_value):
    if pd.isna(ht_value) or ht_value == '':
        return ht_value
    ht_limpo = re.sub(r'\([^)]*\)', '', ht_value).strip()
    return ht_limpo if ht_limpo != '' else ht_value


def extrair_dados_competicao(url, nome_competicao):
    headers = {
        'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36'
    }
    try:
        response = requests.get(url, headers=headers, timeout=10)
        response.raise_for_status()
        soup = BeautifulSoup(response.content, 'html.parser')
        linhas = soup.find_all('tr', class_='odd')
        dados_competicao = []
        for linha in linhas:
            celulas = linha.find_all('td')
            if len(celulas) >= 7:
                data_ingles = celulas[0].get_text(strip=True)
                data_portugues = traduzir_data(data_ingles)
                ft_result = celulas[2].get_text(strip=True)
                if (any(dia in data_portugues for dia in ['Seg', 'Ter', 'Qua', 'Qui', 'Sex', 'Sáb', 'Dom'])
                        and any(caractere.isdigit() for caractere in data_portugues)
                        and 'pp.' not in ft_result):
                    jogo = {
                        'Data': data_portugues,
                        'Time Casa': celulas[1].get_text(strip=True),
                        'Time Visitante': celulas[3].get_text(strip=True),
                        'HT': celulas[5].get_text(strip=True) if len(celulas) > 5 else '',
                        'FT': ft_result,
                        'Competição': nome_competicao
                    }
                    dados_competicao.append(jogo)
        return dados_competicao
    except Exception as e:
        st.warning(f"⚠️ Erro em {nome_competicao}: {str(e)}")
        return []


def extrair_todas_competicoes():
    COMPETICOES = {
        "Brasil Série A": "https://www.soccerstats.com/results.asp?league=brazil&pmtype=bydate",
        "Brasil Série B": "https://www.soccerstats.com/results.asp?league=brazil2&pmtype=bydate",
        "Áustria": "https://www.soccerstats.com/results.asp?league=austria&pmtype=bydate",
        "Argentina": "https://www.soccerstats.com/results.asp?league=argentina&pmtype=bydate",
        "Argentina D2": "https://www.soccerstats.com/results.asp?league=argentina2&pmtype=bydate",
        "Bélgica": "https://www.soccerstats.com/results.asp?league=belgium&pmtype=bydate",
        "Austrália": "https://www.soccerstats.com/results.asp?league=australia&pmtype=bydate",
        "Suíça": "https://www.soccerstats.com/results.asp?league=switzerland&pmtype=bydate",
        "República Tcheca": "https://www.soccerstats.com/results.asp?league=czechrepublic&pmtype=bydate",
        "Alemanha": "https://www.soccerstats.com/results.asp?league=germany&pmtype=bydate",
        "Alemanha D2": "https://www.soccerstats.com/results.asp?league=germany2&pmtype=bydate",
        "Alemanha D3": "https://www.soccerstats.com/results.asp?league=germany3&pmtype=bydate",
        "Dinamarca": "https://www.soccerstats.com/results.asp?league=denmark&pmtype=bydate",
        "Inglaterra": "https://www.soccerstats.com/results.asp?league=england&pmtype=bydate",
        "Inglaterra D2": "https://www.soccerstats.com/results.asp?league=england2&pmtype=bydate",
        "Inglaterra D3": "https://www.soccerstats.com/results.asp?league=england3&pmtype=bydate",
        "Inglaterra D4": "https://www.soccerstats.com/results.asp?league=england4&pmtype=bydate",
        "Inglaterra D5": "https://www.soccerstats.com/results.asp?league=england5&pmtype=bydate",
        "Inglaterra D15": "https://www.soccerstats.com/results.asp?league=england15&pmtype=bydate",
        "Espanha": "https://www.soccerstats.com/results.asp?league=spain&pmtype=bydate",
        "Espanha D2": "https://www.soccerstats.com/results.asp?league=spain2&pmtype=bydate",
        "França": "https://www.soccerstats.com/results.asp?league=france&pmtype=bydate",
        "França D2": "https://www.soccerstats.com/results.asp?league=france2&pmtype=bydate",
        "Grécia": "https://www.soccerstats.com/results.asp?league=greece&pmtype=bydate",
        "Holanda": "https://www.soccerstats.com/results.asp?league=netherlands&pmtype=bydate",
        "Holanda D2": "https://www.soccerstats.com/results.asp?league=netherlands2&pmtype=bydate",
        "Itália": "https://www.soccerstats.com/results.asp?league=italy&pmtype=bydate",
        "Itália D2": "https://www.soccerstats.com/results.asp?league=italy2&pmtype=bydate",
        "Japão": "https://www.soccerstats.com/results.asp?league=japan&pmtype=bydate",
        "Noruega": "https://www.soccerstats.com/results.asp?league=norway&pmtype=bydate",
        "Polônia": "https://www.soccerstats.com/results.asp?league=poland&pmtype=bydate",
        "Portugal": "https://www.soccerstats.com/results.asp?league=portugal&pmtype=bydate",
        "Portugal D2": "https://www.soccerstats.com/results.asp?league=portugal2&pmtype=bydate",
        "Escócia": "https://www.soccerstats.com/results.asp?league=scotland&pmtype=bydate",
        "Escócia D2": "https://www.soccerstats.com/results.asp?league=scotland2&pmtype=bydate",
        "Suécia": "https://www.soccerstats.com/results.asp?league=sweden&pmtype=bydate",
        "Turquia": "https://www.soccerstats.com/results.asp?league=turkey&pmtype=bydate",
        "EUA MLS": "https://www.soccerstats.com/results.asp?league=usa&pmtype=bydate",
        "EUA D2": "https://www.soccerstats.com/results.asp?league=usa2&pmtype=bydate",
        "Canadá": "https://www.soccerstats.com/results.asp?league=canada&pmtype=bydate",
        "Chile": "https://www.soccerstats.com/results.asp?league=chile&pmtype=bydate"
    }
    todos_dados = []
    with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
        futures = {
            executor.submit(extrair_dados_competicao, url, nome): nome
            for nome, url in COMPETICOES.items()
        }
        for future in concurrent.futures.as_completed(futures):
            dados = future.result()
            if dados:
                todos_dados.extend(dados)
    return todos_dados


def obter_data_por_dias(dias):
    data_alvo = datetime.now() + timedelta(days=dias)
    dias_semana_pt = {
        0: 'Seg', 1: 'Ter', 2: 'Qua', 3: 'Qui',
        4: 'Sex', 5: 'Sáb', 6: 'Dom'
    }
    meses_pt = {
        1: 'Jan', 2: 'Fev', 3: 'Mar', 4: 'Abr',
        5: 'Mai', 6: 'Jun', 7: 'Jul', 8: 'Ago',
        9: 'Set', 10: 'Out', 11: 'Nov', 12: 'Dez'
    }
    dia_semana = dias_semana_pt[data_alvo.weekday()]
    dia_mes = data_alvo.day
    mes = meses_pt[data_alvo.month]
    return f"{dia_semana} {dia_mes} {mes}"


def extrair_mes_ano(data_str):
    try:
        partes = data_str.split()
        if len(partes) >= 3:
            mes_pt = partes[2]
            ano = datetime.now().year
            meses_para_numero = {
                'Jan': 1, 'Fev': 2, 'Mar': 3, 'Abr': 4,
                'Mai': 5, 'Jun': 6, 'Jul': 7, 'Ago': 8,
                'Set': 9, 'Out': 10, 'Nov': 11, 'Dez': 12
            }
            mes_numero = meses_para_numero.get(mes_pt, 0)
            if mes_numero == 1 and datetime.now().month == 12:
                ano += 1
            elif mes_numero == 12 and datetime.now().month == 1:
                ano -= 1
            return f"{mes_pt} {ano}"
        return "Desconhecido"
    except:
        return "Desconhecido"


def ordenar_meses(mes_ano_str):
    try:
        if mes_ano_str == "Desconhecido":
            return (0, 0)
        mes, ano = mes_ano_str.split()
        meses_para_numero = {
            'Jan': 1, 'Fev': 2, 'Mar': 3, 'Abr': 4,
            'Mai': 5, 'Jun': 6, 'Jul': 7, 'Ago': 8,
            'Set': 9, 'Out': 10, 'Nov': 11, 'Dez': 12
        }
        mes_numero = meses_para_numero.get(mes, 0)
        ano_numero = int(ano)
        return (ano_numero, mes_numero)
    except:
        return (0, 0)


# 🔥 CONFIGURAÇÃO DA PÁGINA STREAMLIT
st.set_page_config(
    page_title="FutAlgorithm",
    page_icon="⚽",
    layout="wide"
)

# CSS personalizado ATUALIZADO
st.markdown("""
<style>
    .main {
        background-color: #0e1117;
    }
    .stApp {
        background: linear-gradient(135deg, #0c0f15 0%, #1a1d2e 100%);
    }
    .main-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 2rem;
        border-radius: 15px;
        margin-bottom: 2rem;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        border: 1px solid rgba(255, 255, 255, 0.1);
    }
    .main-header h1 {
        color: white;
        text-align: center;
        font-size: 2.8rem;
        font-weight: 800;
        margin-bottom: 0;
        text-shadow: 2px 2px 4px rgba(0, 0, 0, 0.5);
        letter-spacing: 1px;
    }
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        padding: 0 8px;
    }
    .stTabs [data-baseweb="tab"] {
        height: 60px;
        white-space: pre-wrap;
        background: linear-gradient(135deg, #2d3746 0%, #3a4556 100%);
        border-radius: 12px 12px 0 0;
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-bottom: none;
        padding: 15px 25px;
        font-weight: 600;
        font-size: 1rem;
        color: #b0b7c3;
        transition: all 0.3s ease;
        margin: 0 2px;
    }
    .stTabs [data-baseweb="tab"]:hover {
        background: linear-gradient(135deg, #3a4556 0%, #4a5568 100%);
        color: white;
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
    }
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: 1px solid rgba(255, 255, 255, 0.2);
        border-bottom: none;
        box-shadow: 0 4px 20px rgba(102, 126, 234, 0.4);
    }
    .stTabs [data-baseweb="tab-panel"] {
        background: #1a1d2e;
        border-radius: 0 15px 15px 15px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        padding: 2rem;
        margin-top: -1px;
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
    }
    .dica-card {
        background: linear-gradient(135deg, #2d3746 0%, #3a4556 100%);
        border: 1px solid #4a5568;
        border-radius: 12px;
        padding: 15px;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    .dica-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        border-color: #667eea;
    }
    .dica-header {
        font-size: 1.1rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 8px;
    }
    .dica-probabilidade {
        font-size: 1.3rem;
        font-weight: 800;
        color: #48bb78;
        text-align: center;
        margin: 5px 0;
    }
    .dica-stats {
        font-size: 0.9rem;
        color: #b0b7c3;
        text-align: center;
    }
    .ranking-header {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        padding: 15px;
        border-radius: 10px;
        margin: 15px 0;
        text-align: center;
    }
    .liga-card {
        background: linear-gradient(135deg, #2d3746 0%, #3a4556 100%);
        border: 1px solid #4a5568;
        border-radius: 12px;
        padding: 20px;
        margin: 10px 0;
        transition: all 0.3s ease;
    }
    .liga-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 15px rgba(0, 0, 0, 0.3);
        border-color: #667eea;
    }
    .liga-header {
        font-size: 1.2rem;
        font-weight: 700;
        color: #e2e8f0;
        margin-bottom: 8px;
        text-align: center;
    }
    .liga-probabilidade {
        font-size: 1.8rem;
        font-weight: 800;
        color: #48bb78;
        text-align: center;
        margin: 10px 0;
    }
    .liga-stats {
        font-size: 0.9rem;
        color: #b0b7c3;
        text-align: center;
    }
    div[data-testid="stDataFrame"] table {
        width: 100%;
        background: #1e2235;
        border-radius: 10px;
        overflow: hidden;
    }
    div[data-testid="stDataFrame"] th,
    div[data-testid="stDataFrame"] td {
        text-align: center !important;
        vertical-align: middle !important;
        padding: 12px 15px !important;
        border: 1px solid #2d3746 !important;
    }
    div[data-testid="stDataFrame"] thead th {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        font-weight: 700 !important;
        text-align: center !important;
        border: none !important;
        font-size: 1rem;
        text-transform: uppercase;
        letter-spacing: 0.5px;
    }
    div[data-testid="stDataFrame"] tbody tr:nth-child(even) {
        background-color: #1e2235 !important;
    }
    div[data-testid="stDataFrame"] tbody tr:nth-child(odd) {
        background-color: #252a41 !important;
    }
    div[data-testid="stDataFrame"] tbody tr:hover {
        background-color: #2d3746 !important;
        transform: scale(1.01);
        transition: all 0.2s ease;
    }
    div[data-testid="stDataFrame"] td {
        color: #e2e8f0 !important;
        font-weight: 500;
    }
</style>
""", unsafe_allow_html=True)

# Header personalizado
st.markdown("""
<div class="main-header">
    <h1>FutAlgorithm</h1>
</div>
""", unsafe_allow_html=True)

# 🔥 EXECUÇÃO PRINCIPAL MODIFICADA - NOVA SEQUÊNCIA DE ABAS
with st.spinner("🔄 Coletando dados de 40 competições em tempo real..."):
    dados_todos = extrair_todas_competicoes()

if dados_todos:
    df = pd.DataFrame(dados_todos)

    # VERIFICAÇÃO FINAL - Garantir que não há "pp." na coluna FT
    jogos_com_pp = df[df['FT'].str.contains('pp.', na=False)]
    if not jogos_com_pp.empty:
        df = df[~df['FT'].str.contains('pp.', na=False)]

    # 🔥 NOVA SEQUÊNCIA DE ABAS
    tab1, tab2, tab3, tab4 = st.tabs(
        ["🔍 BUSCAR JOGOS", "🎯 ALERTAS INTELIGENTES", "📊 DICAS ESTATÍSTICAS", "🗃️ BASE DE DADOS"])

    with tab1:
        # Aba "Buscar Jogos" - Partidas com coluna "HT" vazia
        df_jogos = df[df['HT'].isna() | (df['HT'] == '')]

        if not df_jogos.empty:
            col1, col2, col3 = st.columns(3)

            with col1:
                competicoes_jogos = ["Todas"] + sorted(df_jogos['Competição'].unique())
                competicao_selecionada_jogos = st.selectbox("Filtrar por competição:", competicoes_jogos,
                                                            key="comp_jogos")

            with col2:
                todos_times_jogos = pd.unique(df_jogos[['Time Casa', 'Time Visitante']].values.ravel('K'))
                time_selecionado_jogos = st.selectbox("Filtrar por time:", ["Todos"] + sorted(todos_times_jogos),
                                                      key="time_jogos")

            with col3:
                df_jogos_copy = df_jogos.copy()
                df_jogos_copy.loc[:, 'Mes_Ano'] = df_jogos_copy['Data'].apply(extrair_mes_ano)
                meses_jogos = ["Todos os Meses"] + sorted(df_jogos_copy['Mes_Ano'].unique(),
                                                          key=ordenar_meses)
                mes_selecionado_jogos = st.selectbox("Filtrar por mês:", meses_jogos, key="mes_jogos")

            # Aplicar filtros - ABA BUSCAR JOGOS
            df_jogos_filtrado = df_jogos.copy()

            if competicao_selecionada_jogos != "Todas":
                df_jogos_filtrado = df_jogos_filtrado[df_jogos_filtrado['Competição'] == competicao_selecionada_jogos]

            if time_selecionado_jogos != "Todos":
                df_jogos_filtrado = df_jogos_filtrado[
                    (df_jogos_filtrado['Time Casa'] == time_selecionado_jogos) |
                    (df_jogos_filtrado['Time Visitante'] == time_selecionado_jogos)
                    ]

            if mes_selecionado_jogos != "Todos os Meses":
                df_jogos_filtrado_copy = df_jogos_filtrado.copy()
                df_jogos_filtrado_copy.loc[:, 'Mes_Ano'] = df_jogos_filtrado_copy['Data'].apply(extrair_mes_ano)
                df_jogos_filtrado = df_jogos_filtrado_copy[df_jogos_filtrado_copy['Mes_Ano'] == mes_selecionado_jogos]

            # 🔥 SELEÇÃO DE PERÍODO
            st.markdown("---")
            st.markdown("### 📅 Selecionar Período")

            col_periodo1, col_periodo2 = st.columns(2)

            with col_periodo1:
                if st.button("🟢 **Próximos 3 Dias**", use_container_width=True, key="btn_3_dias"):
                    periodo_selecionado = "Próximos 3 Dias"
                    st.session_state.periodo = "Próximos 3 Dias"

            with col_periodo2:
                if st.button("🟢 **Próximos 7 Dias**", use_container_width=True, key="btn_7_dias"):
                    periodo_selecionado = "Próximos 7 Dias"
                    st.session_state.periodo = "Próximos 7 Dias"

            if 'periodo' not in st.session_state:
                st.session_state.periodo = "Próximos 3 Dias"

            periodo_selecionado = st.session_state.periodo
            st.info(f"**Período Selecionado:** {periodo_selecionado}")

            # Aplicar filtro de período
            opcoes_periodo = {
                "Próximos 3 Dias": [0, 1, 2],
                "Próximos 7 Dias": [0, 1, 2, 3, 4, 5, 6]
            }

            dias = opcoes_periodo[periodo_selecionado]
            datas_alvo = [obter_data_por_dias(dia) for dia in dias]
            df_jogos_filtrado_periodo = df_jogos_filtrado[df_jogos_filtrado['Data'].isin(datas_alvo)]

            # 🔥 APLICAR ANÁLISE PICO MÁXIMO
            if not df_jogos_filtrado_periodo.empty:
                st.info("🎯 Aplicando análise Pico Máximo... Isso pode levar alguns minutos")

                # Preparar base histórica para análise
                df_base_historica = df[df['HT'].str.contains('(', regex=False, na=False)].copy()
                df_base_historica_limpo = df_base_historica.copy()
                df_base_historica_limpo.loc[:, 'HT'] = df_base_historica_limpo['HT'].apply(limpar_ht)

                # Renomear colunas para compatibilidade
                df_base_historica_limpo = df_base_historica_limpo.rename(columns={
                    'Time Casa': 'Casa',
                    'Time Visitante': 'Fora'
                })

                # Aplicar análise Pico Máximo
                df_jogos_com_analise = adicionar_analise_pico_maximo(
                    df_jogos_filtrado_periodo.rename(columns={
                        'Time Casa': 'Casa',
                        'Time Visitante': 'Fora'
                    }),
                    df_base_historica_limpo
                )

                # Selecionar e ordenar colunas
                colunas_ordenadas = [
                    'Competição', 'Casa', 'Fora',
                    'Casa Vence', 'Empate', 'Fora Vence',
                    'Gols HT', 'Over 0.5 HT', 'Over 1.5 HT',
                    'Casa Marca HT', 'Fora Marca HT',
                    'Gols FT', 'Over 0.5 FT', 'Over 1.5 FT', 'Over 2.5 FT',
                    'Over 3.5 FT', 'Over 4.5 FT',
                    'Casa Marca 1.5', 'Fora Marca 1.5', 'Btts FT', 'Btts & Over 2.5'
                ]

                # Manter apenas colunas existentes
                colunas_existentes = [col for col in colunas_ordenadas if col in df_jogos_com_analise.columns]
                df_jogos_final = df_jogos_com_analise[colunas_existentes]

                # Ordenar por Competição
                df_jogos_ordenado = df_jogos_final.sort_values(['Competição', 'Casa'])

                # Exibir dataframe
                st.dataframe(
                    df_jogos_ordenado,
                    use_container_width=True,
                    hide_index=True,
                    height=600
                )

                # Download específico para Jogos com Pico Máximo
                csv_jogos = df_jogos_ordenado.to_csv(index=False, encoding='utf-8-sig')
                st.download_button(
                    label=f"📥 Download Jogos Pico Máximo ({len(df_jogos_filtrado_periodo)} jogos)",
                    data=csv_jogos,
                    file_name=f"jogos_pico_maximo_{periodo_selecionado.lower().replace(' ', '_')}_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                    mime="text/csv",
                    key="download_jogos_pico"
                )
            else:
                st.warning("Nenhum jogo encontrado para o período selecionado")
        else:
            st.warning("Nenhum jogo futuro encontrado")

    with tab2:
        # 🔥 ABA: ALERTAS INTELIGENTES - CORRIGIDA E MELHORADA
        st.markdown("### 🎯 ALERTAS INTELIGENTES")
        st.markdown("**Rankings por Mercado - Baseado em Dados Históricos da Temporada**")

        if st.button("🔄 Atualizar Alertas", key="reload_alertas", use_container_width=True):
            st.rerun()

        # Preparar base histórica
        df_base_historica = df[df['HT'].str.contains('(', regex=False, na=False)].copy()
        df_base_historica_limpo = df_base_historica.copy()
        df_base_historica_limpo.loc[:, 'HT'] = df_base_historica_limpo['HT'].apply(limpar_ht)
        df_base_historica_limpo = df_base_historica_limpo.rename(columns={
            'Time Casa': 'Casa', 'Time Visitante': 'Fora'
        })

        if not df_base_historica_limpo.empty:
            # Inicializar analisador
            analisador_alertas = AnalisadorAlertasInteligentes(df_base_historica_limpo)

            # 🔥 FILTROS SIMPLIFICADOS
            col1, col2 = st.columns(2)

            with col1:
                mercados_opcoes = list(analisador_alertas.mercados.keys())
                mercado_selecionado = st.selectbox(
                    "💰 Mercado",
                    mercados_opcoes,
                    key="mercado_alertas"
                )

            with col2:
                competicoes_disponiveis = ["Todas"] + sorted(df_base_historica_limpo['Competição'].unique())
                competicao_selecionada = st.selectbox(
                    "🏆 Competição",
                    competicoes_disponiveis,
                    key="comp_alertas"
                )

            # Calcular rankings
            with st.spinner(f"📊 Calculando ranking para {mercado_selecionado}..."):
                ranking_equipes, ranking_ligas = analisador_alertas.gerar_ranking_mercado(
                    mercado_selecionado, competicao_selecionada if competicao_selecionada != "Todas" else None
                )

            if ranking_ligas:
                # 🔥 HEADER DA LIGA TOP
                liga_top = ranking_ligas[0]
                st.markdown("---")

                emoji_posicao = "🥇"

                st.markdown(f"""
                <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                            padding: 25px; border-radius: 15px; margin: 20px 0; color: white; text-align: center;">
                    <div style="font-size: 2rem; margin-bottom: 10px;">
                        {emoji_posicao} {liga_top['Liga']}
                    </div>
                    <div style="font-size: 1.2rem; margin-bottom: 15px;">
                        {mercado_selecionado} &nbsp; ⭐️ &nbsp; <strong>{liga_top['Taxa']:.1f}%</strong>
                    </div>
                    <div style="font-size: 0.9rem; opacity: 0.9;">
                        Baseado em {liga_top['Jogos']} jogos da temporada
                    </div>
                </div>
                """, unsafe_allow_html=True)

                # 🔥 TOP 10 EQUIPES - TABELA
                if ranking_equipes:
                    st.markdown("---")
                    st.markdown("### 🎖️ TOP 10 EQUIPES")

                    # Criar DataFrame para exibição
                    df_top_equipes = pd.DataFrame(ranking_equipes[:10])  # Top 10

                    # Adicionar coluna de ranking com emojis
                    emojis_ranking = ["🥇", "🥈", "🥉", "4️⃣", "5️⃣", "6️⃣", "7️⃣", "8️⃣", "9️⃣", "🔟"]
                    df_top_equipes.insert(0, 'Rank',
                                          [emojis_ranking[i] if i < len(emojis_ranking) else f"{i + 1}️⃣" for i in
                                           range(len(df_top_equipes))])

                    # Formatar colunas
                    df_display = df_top_equipes[
                        ['Rank', 'Equipe', 'Liga', 'Jogos', 'Acertos', 'Taxa', 'Últimos 5']].copy()
                    df_display['Taxa'] = df_display['Taxa'].apply(lambda x: f"{x:.1f}%")
                    df_display['Acertos'] = df_display['Acertos'].astype(int)

                    # Exibir tabela
                    st.dataframe(
                        df_display,
                        use_container_width=True,
                        height=400,
                        hide_index=True
                    )

                    # 🔥 ESTATÍSTICAS
                    st.markdown("---")
                    st.markdown("### 📊 RESUMO ESTATÍSTICO")

                    col_stats1, col_stats2, col_stats3, col_stats4 = st.columns(4)

                    with col_stats1:
                        st.metric("📈 Total Equipes", len(ranking_equipes))

                    with col_stats2:
                        melhor_equipe = ranking_equipes[0] if ranking_equipes else {}
                        st.metric("🎯 Melhor Equipe",
                                  f"{melhor_equipe.get('Equipe', 'N/A')}"
                                  if melhor_equipe else "N/A")

                    with col_stats3:
                        st.metric("🏆 Melhor Taxa",
                                  f"{melhor_equipe.get('Taxa', 0):.1f}%"
                                  if melhor_equipe else "N/A")

                    with col_stats4:
                        acima_70 = len([e for e in ranking_equipes if e['Taxa'] >= 70])
                        st.metric("🔥 Acima de 70%", f"{acima_70} equipes")

                    # Download
                    csv_alertas = pd.DataFrame(ranking_equipes).to_csv(index=False, encoding='utf-8-sig')
                    st.download_button(
                        label=f"📥 Download Ranking Completo ({len(ranking_equipes)} equipes)",
                        data=csv_alertas,
                        file_name=f"alertas_{mercado_selecionado.replace(' ', '_')}_{datetime.now().strftime('%Y%m%d')}.csv",
                        mime="text/csv",
                        use_container_width=True
                    )
                else:
                    st.info("ℹ️ Nenhuma equipe encontrada para os critérios selecionados.")
            else:
                st.warning("⚠️ Nenhuma liga encontrada. Tente selecionar outro mercado.")
        else:
            st.error("❌ Base histórica vazia para cálculo de alertas")

    with tab3:
        # 🔥 ABA: DICAS ESTATÍSTICAS - CORRIGIDA
        st.markdown("### 📊 DICAS ESTATÍSTICAS - PRÓXIMOS 7 DIAS")

        if st.button("🔄 Atualizar Dicas", key="reload_dicas", use_container_width=True):
            st.rerun()

        # Filtrar jogos dos próximos 7 dias
        datas_7_dias = [obter_data_por_dias(dia) for dia in range(7)]
        df_jogos_7_dias = df[df['HT'].isna() | (df['HT'] == '')]
        df_jogos_7_dias = df_jogos_7_dias[df_jogos_7_dias['Data'].isin(datas_7_dias)]

        if not df_jogos_7_dias.empty:
            # Preparar base histórica
            df_base_historica = df[df['HT'].str.contains('(', regex=False, na=False)].copy()
            df_base_historica_limpo = df_base_historica.copy()
            df_base_historica_limpo.loc[:, 'HT'] = df_base_historica_limpo['HT'].apply(limpar_ht)
            df_base_historica_limpo = df_base_historica_limpo.rename(columns={
                'Time Casa': 'Casa', 'Time Visitante': 'Fora'
            })

            # Inicializar analisador
            analisador_dicas = AnalisadorDicasEstatisticas(df_base_historica_limpo)

            # 🔥 FILTROS PARA DICAS
            col1, col2, col3 = st.columns(3)
            with col1:
                competicoes_dicas = ["Todas"] + sorted(df_jogos_7_dias['Competição'].unique())
                competicao_selecionada_dicas = st.selectbox("Filtrar por competição:", competicoes_dicas,
                                                            key="comp_dicas")

            with col2:
                # 🔥 FILTRO POR MERCADO
                mercados_opcoes = ["Todos"] + [config['nome'] for config in analisador_dicas.mercados_config.values()]
                mercado_selecionado_dicas = st.selectbox("Filtrar por mercado:", mercados_opcoes,
                                                         key="mercado_dicas")

            with col3:
                probabilidade_minima = st.slider(
                    "Probabilidade Mínima:",
                    min_value=60,
                    max_value=90,
                    value=70,
                    help="Mostrar apenas dicas com probabilidade acima deste valor"
                )

            # Aplicar filtro de competição
            df_jogos_analise = df_jogos_7_dias.copy()
            if competicao_selecionada_dicas != "Todas":
                df_jogos_analise = df_jogos_analise[df_jogos_analise['Competição'] == competicao_selecionada_dicas]

            # Gerar dicas
            jogos_com_dicas = []

            if len(df_jogos_analise) > 0:
                progress_bar = st.progress(0)
                total_jogos = len(df_jogos_analise)

                for idx, jogo in df_jogos_analise.iterrows():
                    progresso = min((idx + 1) / total_jogos, 1.0)
                    progress_bar.progress(progresso)

                    dicas = analisador_dicas.gerar_dicas_jogo(jogo['Time Casa'], jogo['Time Visitante'],
                                                              mercado_selecionado_dicas if mercado_selecionado_dicas != "Todos" else None)

                    # Filtrar por probabilidade mínima
                    dicas_filtradas = [dica for dica in dicas if dica['probabilidade'] >= probabilidade_minima]

                    if dicas_filtradas:
                        jogos_com_dicas.append({
                            'data': jogo['Data'],
                            'liga': jogo['Competição'],
                            'casa': jogo['Time Casa'],
                            'fora': jogo['Time Visitante'],
                            'dicas': dicas_filtradas
                        })

                progress_bar.empty()

            # Exibir dicas
            if jogos_com_dicas:
                st.success(f"🎯 {len(jogos_com_dicas)} jogos com dicas estatísticas encontrados!")

                # Ordenar jogos pela maior probabilidade
                jogos_ordenados = sorted(jogos_com_dicas,
                                         key=lambda x: max([d['probabilidade'] for d in x['dicas']]),
                                         reverse=True)

                for jogo in jogos_ordenados:
                    with st.container():
                        st.markdown(f"""
                        <div style="background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); 
                                    padding: 15px; border-radius: 12px; margin: 15px 0; color: white;">
                            <div style="font-size: 16px; font-weight: bold; margin-bottom: 10px;">
                                ⚽ {jogo['casa']} vs {jogo['fora']} | 📅 {jogo['data']} | 🏆 {jogo['liga']}
                            </div>
                        </div>
                        """, unsafe_allow_html=True)

                        # Exibir dicas em colunas
                        colunas_dicas = st.columns(2)
                        for idx, dica in enumerate(jogo['dicas']):
                            with colunas_dicas[idx % 2]:
                                st.markdown(f"""
                                <div class="dica-card">
                                    <div class="dica-header">
                                        {dica['icone']} {dica['mercado']}
                                    </div>
                                    <div class="dica-probabilidade">
                                        {dica['probabilidade']:.1f}%
                                    </div>
                                    <div class="dica-stats">
                                        🏠 {jogo['casa']}: {dica['casa_percent']:.1f}% | 
                                        ✈️ {jogo['fora']}: {dica['fora_percent']:.1f}%
                                    </div>
                                </div>
                                """, unsafe_allow_html=True)
            else:
                st.info("ℹ️ Nenhuma dica estatística encontrada para os critérios selecionados.")
        else:
            st.warning("Nenhum jogo encontrado para os próximos 7 dias")

    with tab4:
        # 🔥 ABA BASE DE DADOS - CORRIGIDA
        st.markdown("### 🗃️ BASE DE DADOS HISTÓRICOS")

        # Filtrar jogos com dados HT completos (jogos já realizados)
        df_base_dados = df[df['HT'].str.contains('(', regex=False, na=False)]

        if not df_base_dados.empty:
            col1, col2, col3 = st.columns(3)

            with col1:
                competicoes_bd = ["Todas"] + sorted(df_base_dados['Competição'].unique())
                competicao_selecionada_bd = st.selectbox("Filtrar por competição:", competicoes_bd, key="comp_bd")

            with col2:
                todos_times_bd = pd.unique(df_base_dados[['Time Casa', 'Time Visitante']].values.ravel('K'))
                time_selecionado_bd = st.selectbox("Filtrar por time:", ["Todos"] + sorted(todos_times_bd),
                                                   key="time_bd")

            with col3:
                df_base_dados_copy = df_base_dados.copy()
                df_base_dados_copy.loc[:, 'Mes_Ano'] = df_base_dados_copy['Data'].apply(extrair_mes_ano)
                meses_bd = ["Todos os Meses"] + sorted(df_base_dados_copy['Mes_Ano'].unique(),
                                                       key=ordenar_meses)
                mes_selecionado_bd = st.selectbox("Filtrar por mês:", meses_bd, key="mes_bd")

            # Aplicar filtros - ABA BASE DE DADOS
            df_base_dados_filtrado = df_base_dados.copy()

            if competicao_selecionada_bd != "Todas":
                df_base_dados_filtrado = df_base_dados_filtrado[
                    df_base_dados_filtrado['Competição'] == competicao_selecionada_bd]

            if time_selecionado_bd != "Todos":
                df_base_dados_filtrado = df_base_dados_filtrado[
                    (df_base_dados_filtrado['Time Casa'] == time_selecionado_bd) |
                    (df_base_dados_filtrado['Time Visitante'] == time_selecionado_bd)
                    ]

            if mes_selecionado_bd != "Todos os Meses":
                df_base_dados_filtrado_copy = df_base_dados_filtrado.copy()
                df_base_dados_filtrado_copy.loc[:, 'Mes_Ano'] = df_base_dados_filtrado_copy['Data'].apply(
                    extrair_mes_ano)
                df_base_dados_filtrado = df_base_dados_filtrado_copy[
                    df_base_dados_filtrado_copy['Mes_Ano'] == mes_selecionado_bd]

            # Aplicar limpeza na coluna HT
            df_base_dados_limpo = df_base_dados_filtrado.copy()
            df_base_dados_limpo.loc[:, 'HT'] = df_base_dados_limpo['HT'].apply(limpar_ht)

            # Selecionar e renomear colunas específicas
            colunas_selecionadas = ['Data', 'Competição', 'Time Casa', 'Time Visitante', 'HT', 'FT']
            df_base_dados_selecionado = df_base_dados_limpo[colunas_selecionadas].copy()

            # Renomear as colunas
            df_base_dados_selecionado = df_base_dados_selecionado.rename(columns={
                'Time Casa': 'Casa',
                'Time Visitante': 'Fora'
            })

            # Ordenar por Data e Competição
            df_base_dados_ordenado = df_base_dados_selecionado.sort_values(['Data', 'Competição'])

            # Exibir dataframe
            st.dataframe(
                df_base_dados_ordenado,
                use_container_width=True,
                hide_index=True,
                height=600
            )

            # Download específico para Base de Dados
            csv_base_dados = df_base_dados_ordenado.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label=f"📥 Download Base de Dados ({len(df_base_dados_filtrado)} jogos)",
                data=csv_base_dados,
                file_name=f"base_dados_{datetime.now().strftime('%Y%m%d_%H%M')}.csv",
                mime="text/csv",
                key="download_base_dados"
            )

            # Estatísticas da base
            st.markdown("---")
            col_stats1, col_stats2, col_stats3 = st.columns(3)
            with col_stats1:
                st.metric("Total de Jogos", len(df_base_dados_filtrado))
            with col_stats2:
                st.metric("Competições", df_base_dados_filtrado['Competição'].nunique())
            with col_stats3:
                st.metric("Times Únicos",
                          pd.unique(df_base_dados_filtrado[['Time Casa', 'Time Visitante']].values.ravel('K')).size)

        else:
            st.warning("Nenhum jogo histórico encontrado na base de dados")

else:
    st.error("❌ Não foi possível extrair os dados. Verifique sua conexão.")
    if st.button("🔄 Tentar Novamente"):
        st.rerun()