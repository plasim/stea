import re
from dataclasses import dataclass
from io import BytesIO
from pathlib import Path

import pandas as pd
import plotly.express as px
import streamlit as st

VALUE_COLS = ['Haettu', 'Ehdotettu', 'Myönnetty']
TEXT_COLS = [
    'Järjestö',
    'Y-tunnus',
    'Avustuksen käyttötarkoitus',
    'Avustuslaji',
    'Maantiet. alue',
    'Avustuskokonaisuus',
    'Järjestöluokka',
]

GROUPINGS: dict[str, list[str]] = {
    'org': ['Järjestö'],
    'org_kokonaisuus': ['Järjestö', 'Avustuskokonaisuus'],
    'org_avustuslaji': ['Järjestö', 'Avustuslaji'],
    'org_kayttotarkoitus': ['Järjestö', 'Avustuksen käyttötarkoitus'],
    'full_row': [
        'Järjestö',
        'Avustuksen käyttötarkoitus',
        'Avustuslaji',
        'Maantiet. alue',
        'Avustuskokonaisuus',
        'Järjestöluokka',
    ],
}

GROUPING_LABELS: dict[str, tuple[str, str]] = {
    'org': ('Järjestö', 'Organization'),
    'org_kokonaisuus': ('Järjestö + Avustuskokonaisuus', 'Organization + Funding category'),
    'org_avustuslaji': ('Järjestö + Avustuslaji', 'Organization + Grant type'),
    'org_kayttotarkoitus': ('Järjestö + Käyttötarkoitus', 'Organization + Purpose'),
    'full_row': ('Täysi rivi (uniikki)', 'Full row (unique)'),
}


@dataclass(frozen=True)
class Filters:
    years: tuple[int, int]
    keyword: str
    keyword_mode: str  # "all" | "any"
    organization: list[str]
    avustuslaji_type: list[str]
    kokonaisuus: list[str]
    jarjestoluokka: list[str]
    geography_contains: str


def _cache_data(*args, **kwargs):
    if __name__ == '__main__':
        return st.cache_data(*args, **kwargs)

    def decorator(func):
        return func

    return decorator


@_cache_data(show_spinner=False)
def load_stea_excel_from_path(path: str) -> pd.DataFrame:
    return _load_stea_excel(pd.ExcelFile(path))


@_cache_data(show_spinner=False)
def load_stea_excel_from_bytes(data: bytes) -> pd.DataFrame:
    return _load_stea_excel(pd.ExcelFile(BytesIO(data)))


def _load_stea_excel(xl: pd.ExcelFile) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for sheet in xl.sheet_names:
        df = xl.parse(sheet)
        df['_sheet'] = str(sheet)
        frames.append(df)

    df = pd.concat(frames, ignore_index=True)

    missing = [c for c in (TEXT_COLS + ['Vuosi'] + VALUE_COLS) if c not in df.columns]
    if missing:
        raise ValueError(f'Missing expected columns: {missing}')

    df['Vuosi'] = pd.to_numeric(df['Vuosi'], errors='coerce').astype('Int64')
    for col in VALUE_COLS:
        df[col] = pd.to_numeric(df[col], errors='coerce').fillna(0.0)

    for col in TEXT_COLS:
        df[col] = df[col].fillna('').astype(str)

    df['Avustuslaji_tyyppi'] = df['Avustuslaji'].str.strip().str.split(r'\s+').str[0].fillna('')
    return df


def normalize_text(text: str) -> str:
    return re.sub(r'\s+', ' ', (text or '').strip())


def apply_filters(df: pd.DataFrame, filters: Filters) -> pd.DataFrame:
    year_min, year_max = filters.years
    out = df[(df['Vuosi'] >= year_min) & (df['Vuosi'] <= year_max)].copy()

    if filters.organization:
        out = out[out['Järjestö'].isin(filters.organization)]
    if filters.kokonaisuus:
        out = out[out['Avustuskokonaisuus'].isin(filters.kokonaisuus)]
    if filters.jarjestoluokka:
        out = out[out['Järjestöluokka'].isin(filters.jarjestoluokka)]
    if filters.avustuslaji_type:
        out = out[out['Avustuslaji_tyyppi'].isin(filters.avustuslaji_type)]

    geo = normalize_text(filters.geography_contains)
    if geo:
        out = out[out['Maantiet. alue'].str.contains(re.escape(geo), case=False, na=False)]

    keyword = normalize_text(filters.keyword)
    if keyword:
        tokens = [t for t in re.split(r'\s+', keyword) if t]
        haystack = (
            out[
                [
                    'Järjestö',
                    'Avustuksen käyttötarkoitus',
                    'Avustuskokonaisuus',
                    'Järjestöluokka',
                    'Maantiet. alue',
                    'Avustuslaji',
                ]
            ]
            .astype(str)
            .agg(' '.join, axis=1)
            .str.lower()
        )
        if filters.keyword_mode == 'any':
            mask = pd.Series(False, index=out.index)
            for token in tokens:
                mask |= haystack.str.contains(re.escape(token.lower()), na=False)
        else:
            mask = pd.Series(True, index=out.index)
            for token in tokens:
                mask &= haystack.str.contains(re.escape(token.lower()), na=False)
        out = out[mask]

    return out


def format_eur(value: float) -> str:
    try:
        return f'{value:,.0f} €'.replace(',', ' ').replace('\xa0', ' ')
    except Exception:
        return str(value)


def build_entry_columns(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    safe = df[group_cols].fillna('').astype(str)
    df = df.copy()
    df['_entry_key'] = safe.agg('\u241f'.join, axis=1)
    df['_entry_label'] = safe.agg(' | '.join, axis=1).map(lambda s: s if len(s) <= 140 else (s[:139] + '…'))
    return df


def group_per_year(df: pd.DataFrame, group_cols: list[str]) -> pd.DataFrame:
    cols = ['Vuosi', '_entry_key', '_entry_label']
    grouped = (
        df.groupby(cols, dropna=False)[VALUE_COLS]
        .sum()
        .reset_index()
        .sort_values(['Vuosi', 'Myönnetty', 'Haettu'], ascending=[True, False, False])
    )
    return grouped


def entry_purpose_summary(df: pd.DataFrame) -> pd.DataFrame:
    purpose_col = 'Avustuksen käyttötarkoitus'
    if purpose_col not in df.columns:
        return pd.DataFrame({'_entry_key': [], '_purpose_summary': []})

    tmp = df[['_entry_key', purpose_col, 'Myönnetty']].copy()
    tmp[purpose_col] = tmp[purpose_col].fillna('').astype(str)
    tmp['Myönnetty'] = pd.to_numeric(tmp['Myönnetty'], errors='coerce').fillna(0.0)

    by = (
        tmp.groupby(['_entry_key', purpose_col], dropna=False)['Myönnetty']
        .sum()
        .reset_index()
        .sort_values(['_entry_key', 'Myönnetty'], ascending=[True, False])
    )
    top = by.drop_duplicates(subset=['_entry_key'], keep='first').rename(columns={purpose_col: '_purpose_summary'})
    return top[['_entry_key', '_purpose_summary']]


def main() -> None:
    st.set_page_config(page_title='STEA dashboard', layout='wide')

    lang_choice = st.sidebar.selectbox('Kieli / Language', options=['Suomi', 'English'], index=0)
    lang = 'fi' if lang_choice == 'Suomi' else 'en'

    def t(fi: str, en: str) -> str:
        return fi if lang == 'fi' else en

    st.title(t('STEA-avustukset', 'STEA funding dashboard'))

    default_path = Path(__file__).with_name('STEA_2026.xlsx')
    file = st.sidebar.file_uploader(t('Excel (.xlsx)', 'Excel (.xlsx)'), type=['xlsx'])
    if file is None and not default_path.exists():
        st.info(
            t(
                f'Laita `{default_path.name}` tämän sovelluksen viereen tai lataa .xlsx vasemman reunan valikosta.',
                f'Place `{default_path.name}` next to this app, or upload an .xlsx in the sidebar.',
            )
        )
        return

    with st.spinner(t('Ladataan Excel…', 'Loading Excel…')):
        if file is not None:
            df = load_stea_excel_from_bytes(file.getvalue())
        else:
            df = load_stea_excel_from_path(str(default_path))

    year_min = int(df['Vuosi'].dropna().min())
    year_max = int(df['Vuosi'].dropna().max())

    st.sidebar.header(t('Suodattimet', 'Filters'))
    years = st.sidebar.slider(
        t('Vuodet', 'Year range'),
        min_value=year_min,
        max_value=year_max,
        value=(year_min, year_max),
    )
    keyword = st.sidebar.text_input(
        t('Vapaasanahaku', 'Keyword search'),
        placeholder=t('esim. nuoret mielenterveys', 'e.g. youth mental health'),
    )
    keyword_mode = st.sidebar.radio(
        t('Hakutapa', 'Search mode'),
        options=['all', 'any'],
        index=0,
        format_func=lambda v: t('Kaikki sanat (AND)', 'All terms (AND)')
        if v == 'all'
        else t('Mikä tahansa sana (OR)', 'Any term (OR)'),
    )
    geography_contains = st.sidebar.text_input(
        t('Maantiet. alue sisältää', 'Geography contains'),
        placeholder=t('esim. Helsinki', 'e.g. Helsinki'),
    )

    org_options = sorted([o for o in df['Järjestö'].unique().tolist() if o])
    organization = st.sidebar.multiselect(t('Järjestö', 'Organization'), options=org_options)

    type_options = sorted([t for t in df['Avustuslaji_tyyppi'].unique().tolist() if t])
    avustuslaji_type = st.sidebar.multiselect(t('Avustuslaji (tyyppi)', 'Grant type (prefix)'), options=type_options)

    kokonaisuus_options = sorted([k for k in df['Avustuskokonaisuus'].unique().tolist() if k])
    kokonaisuus = st.sidebar.multiselect(t('Avustuskokonaisuus', 'Funding category'), options=kokonaisuus_options)

    luokka_options = sorted([k for k in df['Järjestöluokka'].unique().tolist() if k])
    jarjestoluokka = st.sidebar.multiselect(t('Järjestöluokka', 'Organization class'), options=luokka_options)

    filters = Filters(
        years=years,
        keyword=keyword,
        keyword_mode=keyword_mode,
        organization=organization,
        avustuslaji_type=avustuslaji_type,
        kokonaisuus=kokonaisuus,
        jarjestoluokka=jarjestoluokka,
        geography_contains=geography_contains,
    )
    fdf = apply_filters(df, filters)

    st.caption(
        t(
            f'Rivejä: {len(fdf):,} (yhteensä {len(df):,})',
            f'Rows: {len(fdf):,} (from {len(df):,})',
        ).replace(',', ' ')
    )
    if fdf.empty:
        st.warning(t('Ei tuloksia nykyisillä suodattimilla.', 'No rows match the current filters.'))
        return

    kpi1, kpi2, kpi3, kpi4 = st.columns(4)
    kpi1.metric(t('Haettu (summa)', 'Applied (sum)'), format_eur(float(fdf['Haettu'].sum())))
    kpi2.metric(t('Ehdotettu (summa)', 'Proposed (sum)'), format_eur(float(fdf['Ehdotettu'].sum())))
    kpi3.metric(t('Myönnetty (summa)', 'Approved (sum)'), format_eur(float(fdf['Myönnetty'].sum())))
    kpi4.metric(t('Järjestöjä', 'Organizations'), int(fdf['Järjestö'].nunique()))

    per_year = fdf.groupby('Vuosi')[VALUE_COLS].sum().reset_index().sort_values('Vuosi')
    per_year_long = per_year.melt(id_vars=['Vuosi'], value_vars=VALUE_COLS, var_name='Tyyppi', value_name='Euroa')
    fig_year = px.bar(
        per_year_long,
        x='Vuosi',
        y='Euroa',
        color='Tyyppi',
        barmode='group',
        title=t('Summat per vuosi (suodatettu)', 'Totals per year (filtered)'),
        labels={'Euroa': '€'},
    )
    fig_year.update_layout(legend_title_text='')
    st.plotly_chart(fig_year, use_container_width=True)

    st.subheader(t('Tarkastelu: uniikit rivit', 'Per-entry view'))
    group_choice = st.selectbox(
        t('Uniikin rivin määritelmä', 'Unique entry definition'),
        options=list(GROUPINGS.keys()),
        index=0,
        format_func=lambda k: t(*GROUPING_LABELS.get(k, (k, k))),
    )
    group_cols = GROUPINGS[group_choice]
    df_entries = build_entry_columns(fdf, group_cols)

    top_n = st.slider(
        t('Top N (jos ei valita rivejä)', 'Top N entries (when not selecting entries)'),
        min_value=5,
        max_value=50,
        value=15,
        step=5,
    )
    metric_choice = st.selectbox(
        t('Järjestä', 'Rank by'),
        options=['Myönnetty', 'Haettu', 'Ehdotettu'],
        index=0,
        format_func=lambda k: {
            'Myönnetty': t('Myönnetty', 'Approved'),
            'Haettu': t('Haettu', 'Applied'),
            'Ehdotettu': t('Ehdotettu', 'Proposed'),
        }.get(k, k),
    )

    entry_options = (
        df_entries[['_entry_key', '_entry_label']]
        .drop_duplicates()
        .sort_values('_entry_label')['_entry_label']
        .tolist()
    )
    selected_labels = st.multiselect(
        t('Valitse rivit (valinnainen)', 'Select entries (optional)'),
        options=entry_options,
        help=t(
            'Jätä tyhjäksi, niin näytetään Top N valitun mittarin mukaan.',
            'Leave empty to show the Top N entries by the selected metric.',
        ),
    )

    if selected_labels:
        selected_keys = set(
            df_entries[df_entries['_entry_label'].isin(selected_labels)]['_entry_key'].unique().tolist()
        )
        df_entries = df_entries[df_entries['_entry_key'].isin(selected_keys)]
    else:
        totals = (
            df_entries.groupby(['_entry_key', '_entry_label'], dropna=False)[metric_choice]
            .sum()
            .reset_index()
            .sort_values(metric_choice, ascending=False)
            .head(top_n)
        )
        df_entries = df_entries[df_entries['_entry_key'].isin(set(totals['_entry_key'].tolist()))]

    grouped = group_per_year(df_entries, group_cols).merge(
        entry_purpose_summary(df_entries), on='_entry_key', how='left'
    )
    purpose_fallback = t('(ei määritelty)', '(not specified)')
    grouped['_purpose_summary'] = grouped['_purpose_summary'].fillna('').replace({'': purpose_fallback})

    fig_entries = px.line(
        grouped,
        x='Vuosi',
        y='Myönnetty',
        color='_entry_label',
        line_group='_entry_label',
        markers=True,
        custom_data=['_purpose_summary'],
        title=t('Myönnetty per vuosi (uniikit rivit)', 'Approved per year (entries)'),
        labels={'_entry_label': 'Entry', 'Myönnetty': '€'},
    )
    purpose_label = t('Käyttötarkoitus', 'Purpose')
    value_label = t('Myönnetty', 'Approved')
    fig_entries.update_traces(
        hovertemplate=f'<b>%{{fullData.name}}</b><br>{purpose_label}: %{{customdata[0]}}<br>{value_label}: %{{y:,.0f}} €<extra></extra>'
    )
    fig_entries.update_layout(legend_title_text='')
    st.plotly_chart(fig_entries, use_container_width=True)

    st.subheader(t('Taulukko', 'Data table'))
    show_cols = [
        'Vuosi',
        'Järjestö',
        'Y-tunnus',
        'Avustuskokonaisuus',
        'Järjestöluokka',
        'Avustuslaji',
        'Maantiet. alue',
        'Avustuksen käyttötarkoitus',
        'Haettu',
        'Ehdotettu',
        'Myönnetty',
    ]
    show_cols = [c for c in show_cols if c in fdf.columns]
    st.dataframe(fdf[show_cols], use_container_width=True, height=420)

    csv = fdf[show_cols].to_csv(index=False).encode('utf-8')
    st.download_button(
        t('Lataa suodatettu CSV', 'Download filtered CSV'),
        data=csv,
        file_name='stea_filtered.csv',
        mime='text/csv',
    )


if __name__ == '__main__':
    main()
