//! LXCat/BOLSIG+形式の断面積データと、その読み込み。
//!
//! 対応する書式:
//! - 1行目のキーワード: ELASTIC, EFFECTIVE, EXCITATION, IONIZATION, ATTACHMENT, ROTATION
//! - 2行目の反応式: `A`、`A -> B`、`A <-> B`（`<->`は逆過程を有効にする）
//! - 3行目: ELASTIC/EFFECTIVEは質量比、EXCITATIONは「しきい値 [統計重み比]」、
//!   IONIZATIONはしきい値。ROTATIONは3行目と4行目に下準位・上準位の「エネルギー 統計重み」。
//! - 表: 2列（エネルギー eV、断面積 m²）。3列目があれば運動量移行断面積として読む
//!   （このときの2列目は積分断面積）。

use std::{fmt, fs, path::Path};

use crate::interp::interp;

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
pub enum Kind {
    Elastic,
    Effective,
    Excitation,
    Ionization,
    Attachment,
    Rotation,
}

impl Kind {
    pub const ALL: [Kind; 6] = [
        Kind::Elastic,
        Kind::Effective,
        Kind::Excitation,
        Kind::Ionization,
        Kind::Attachment,
        Kind::Rotation,
    ];

    pub fn keyword(self) -> &'static str {
        match self {
            Kind::Elastic => "ELASTIC",
            Kind::Effective => "EFFECTIVE",
            Kind::Excitation => "EXCITATION",
            Kind::Ionization => "IONIZATION",
            Kind::Attachment => "ATTACHMENT",
            Kind::Rotation => "ROTATION",
        }
    }

    /// ファイル中のキーワード（大文字、前後空白なし）に一致するときだけ返す。
    pub fn from_keyword(value: &str) -> Option<Self> {
        Self::ALL.into_iter().find(|kind| kind.keyword() == value)
    }

    /// API用。大文字小文字を区別しない。
    pub fn parse(value: &str) -> Result<Self, String> {
        let upper = value.trim().to_ascii_uppercase();
        Self::from_keyword(&upper).ok_or_else(|| format!("unknown cross-section kind: {value:?}"))
    }
}

/// ROTATIONの準位（基底状態からのエネルギーと統計重み）。
#[derive(Clone, Copy, Debug, PartialEq)]
pub struct LevelState {
    pub energy_ev: f64,
    pub weight: f64,
}

/// 非減少のエネルギー格子上の表。範囲外は下側0、上側は最後の値で一定。
#[derive(Clone, Debug, PartialEq)]
pub struct Table {
    pub energy: Vec<f64>,
    pub values: Vec<f64>,
}

impl Table {
    pub fn new(energy: Vec<f64>, values: Vec<f64>) -> Result<Self, String> {
        if energy.len() != values.len() {
            return Err(format!(
                "table has {} energies but {} values",
                energy.len(),
                values.len()
            ));
        }
        if energy.is_empty() {
            return Err("cross-section data must not be empty".into());
        }
        if energy.iter().chain(&values).any(|value| !value.is_finite()) {
            return Err("cross-section table must contain only finite values".into());
        }
        if values.iter().any(|value| *value < 0.0) {
            return Err("cross sections must be non-negative".into());
        }
        if let Some(index) = energy.windows(2).position(|pair| pair[1] < pair[0]) {
            return Err(format!(
                "energies must be non-decreasing ({} after {})",
                energy[index + 1],
                energy[index]
            ));
        }
        Ok(Self { energy, values })
    }

    pub fn at(&self, energy_ev: f64) -> f64 {
        let right = *self.values.last().expect("validated non-empty table");
        interp(energy_ev, &self.energy, &self.values, 0.0, right)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct CrossSection {
    pub kind: Kind,
    /// 2行目の反応式（例: `Ar`, `Ar -> Ar*`, `HF <-> HF(v1)`）。
    pub species: String,
    pub name: String,
    pub threshold_ev: f64,
    pub mass_ratio: Option<f64>,
    /// 上準位と下準位の統計重み比 g_up/g_low（EXCITATION）。
    pub weight_ratio: Option<f64>,
    pub lower_state: Option<LevelState>,
    pub upper_state: Option<LevelState>,
    /// 衝突頻度・エネルギー損失・速度係数に使う断面積。
    pub table: Table,
    /// 運動量移行断面積。与えられたときは角度分布の異方性に使う。
    pub momentum_transfer: Option<Table>,
    pub comment: String,
}

impl CrossSection {
    pub fn validate(&self) -> Result<(), String> {
        let label = &self.name;
        if !self.threshold_ev.is_finite() {
            return Err(format!("{label}: threshold must be finite"));
        }
        if let Some(ratio) = self.mass_ratio
            && (!ratio.is_finite() || ratio <= 0.0 || ratio >= 1.0)
        {
            return Err(format!(
                "{label}: mass ratio must be in (0, 1), got {ratio}"
            ));
        }
        if let Some(weight) = self.weight_ratio
            && (!weight.is_finite() || weight <= 0.0)
        {
            return Err(format!(
                "{label}: statistical weight ratio must be positive, got {weight}"
            ));
        }
        match self.kind {
            Kind::Rotation => {
                let (Some(lower), Some(upper)) = (self.lower_state, self.upper_state) else {
                    return Err(format!("{label}: ROTATION needs lower and upper states"));
                };
                for state in [lower, upper] {
                    if !state.energy_ev.is_finite()
                        || !state.weight.is_finite()
                        || state.weight <= 0.0
                    {
                        return Err(format!("{label}: invalid rotational state {state:?}"));
                    }
                }
                let gap = upper.energy_ev - lower.energy_ev;
                if gap <= 0.0 {
                    return Err(format!(
                        "{label}: upper state must lie above the lower state"
                    ));
                }
                if (self.threshold_ev - gap).abs() > 1e-12 * gap.max(1.0) {
                    return Err(format!(
                        "{label}: threshold must equal the state energy difference"
                    ));
                }
            }
            Kind::Excitation | Kind::Ionization if self.threshold_ev < 0.0 => {
                return Err(format!("{label}: threshold must be non-negative"));
            }
            _ => {}
        }
        Ok(())
    }

    /// しきい値未満は0とした断面積。
    pub fn sigma_at(&self, energy_ev: f64) -> f64 {
        mask_threshold(self.table.at(energy_ev), energy_ev, self.threshold_ev)
    }

    pub fn momentum_transfer_at(&self, energy_ev: f64) -> Option<f64> {
        self.momentum_transfer
            .as_ref()
            .map(|table| mask_threshold(table.at(energy_ev), energy_ev, self.threshold_ev))
    }

    /// 反応式の左辺（標的）。
    pub fn target(&self) -> &str {
        split_reaction(&self.species).0
    }

    /// 反応式の右辺（生成物）。矢印がなければ`None`。
    pub fn product(&self) -> Option<&str> {
        split_reaction(&self.species).1
    }

    /// `<->`で逆過程が指定されているか。
    pub fn is_reversible(&self) -> bool {
        split_reaction(&self.species).2
    }
}

fn mask_threshold(value: f64, energy_ev: f64, threshold_ev: f64) -> f64 {
    if threshold_ev > 0.0 && energy_ev < threshold_ev {
        0.0
    } else {
        value
    }
}

fn split_reaction(species: &str) -> (&str, Option<&str>, bool) {
    if let Some((left, right)) = species.split_once("<->") {
        (left.trim(), Some(right.trim()), true)
    } else if let Some((left, right)) = species.split_once("->") {
        (left.trim(), Some(right.trim()), false)
    } else {
        (species.trim(), None, false)
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ParseError {
    /// 1始まりの行番号。
    pub line: usize,
    pub message: String,
}

impl fmt::Display for ParseError {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        write!(f, "line {}: {}", self.line, self.message)
    }
}

impl std::error::Error for ParseError {}

fn error(line: usize, message: impl Into<String>) -> ParseError {
    ParseError {
        line,
        message: message.into(),
    }
}

fn is_dashed(line: &str) -> bool {
    let trimmed = line.trim();
    trimmed.len() >= 5 && trimmed.bytes().all(|byte| byte == b'-')
}

/// パラメータ行の数値。`!`や`#`以降はコメントとして無視する。
fn numeric_tokens(line: &str) -> Vec<Result<f64, String>> {
    line.split_whitespace()
        .take_while(|token| !token.starts_with('!') && !token.starts_with('#'))
        .map(|token| {
            token
                .parse::<f64>()
                .map_err(|_| token.to_string())
                .and_then(|value| {
                    if value.is_finite() {
                        Ok(value)
                    } else {
                        Err(token.to_string())
                    }
                })
        })
        .collect()
}

fn strip_prefix_ignore_case<'a>(line: &'a str, prefix: &str) -> Option<&'a str> {
    let head = line.get(..prefix.len())?;
    head.eq_ignore_ascii_case(prefix)
        .then(|| &line[prefix.len()..])
}

/// ファイルを読む。UTF-8（BOM可）で読めないときはLatin-1として読む。
pub fn parse_file(path: impl AsRef<Path>) -> Result<Vec<CrossSection>, String> {
    let path = path.as_ref();
    let bytes = fs::read(path).map_err(|err| format!("{}: {err}", path.display()))?;
    let text = match String::from_utf8(bytes) {
        Ok(text) => text,
        Err(err) => err
            .into_bytes()
            .iter()
            .map(|byte| char::from(*byte))
            .collect(),
    };
    parse_str(&text).map_err(|err| format!("{}: {err}", path.display()))
}

pub fn parse_str(text: &str) -> Result<Vec<CrossSection>, ParseError> {
    let text = text.strip_prefix('\u{feff}').unwrap_or(text);
    let lines: Vec<&str> = text.lines().collect();
    let mut sections = Vec::new();
    let mut index = 0;
    while index < lines.len() {
        let Some(kind) = Kind::from_keyword(lines[index].trim()) else {
            index += 1;
            continue;
        };
        let keyword_line = index + 1;
        index += 1;
        let species = lines
            .get(index)
            .map(|line| line.trim())
            .filter(|line| !line.is_empty() && !is_dashed(line))
            .ok_or_else(|| {
                error(
                    keyword_line,
                    format!("{} block without a species line", kind.keyword()),
                )
            })?
            .to_string();
        index += 1;

        let mut threshold_ev = 0.0;
        let mut mass_ratio = None;
        let mut weight_ratio = None;
        let mut states = [None, None];
        match kind {
            Kind::Attachment => {}
            Kind::Elastic | Kind::Effective => {
                // 質量比の行は省略可（Gasの分子量で補う）
                if let Some(Ok(value)) = lines
                    .get(index)
                    .and_then(|line| numeric_tokens(line).into_iter().next())
                {
                    mass_ratio = Some(value);
                    index += 1;
                }
            }
            Kind::Excitation | Kind::Ionization => {
                let line_number = index + 1;
                let line = lines.get(index).copied().unwrap_or("");
                let tokens = numeric_tokens(line);
                match tokens.first() {
                    Some(Ok(value)) => threshold_ev = *value,
                    _ => {
                        return Err(error(
                            line_number,
                            format!(
                                "{} block for {species:?} needs the threshold energy on line 3, found {:?}",
                                kind.keyword(),
                                line.trim()
                            ),
                        ));
                    }
                }
                if kind == Kind::Excitation {
                    match tokens.get(1) {
                        Some(Ok(value)) => weight_ratio = Some(*value),
                        Some(Err(token)) => {
                            return Err(error(
                                line_number,
                                format!("cannot read the statistical weight ratio {token:?}"),
                            ));
                        }
                        None => {}
                    }
                }
                index += 1;
            }
            Kind::Rotation => {
                for (slot, label) in states.iter_mut().zip(["lower", "upper"]) {
                    let line_number = index + 1;
                    let line = lines.get(index).copied().unwrap_or("");
                    let tokens = numeric_tokens(line);
                    match (tokens.first(), tokens.get(1)) {
                        (Some(Ok(energy_ev)), Some(Ok(weight))) => {
                            *slot = Some(LevelState {
                                energy_ev: *energy_ev,
                                weight: *weight,
                            });
                        }
                        _ => {
                            return Err(error(
                                line_number,
                                format!(
                                    "ROTATION block for {species:?} needs the {label} state as 'energy weight', found {:?}",
                                    line.trim()
                                ),
                            ));
                        }
                    }
                    index += 1;
                }
                if let [Some(lower), Some(upper)] = states {
                    threshold_ev = upper.energy_ev - lower.energy_ev;
                }
            }
        }

        let mut name = None;
        let mut comments = Vec::new();
        loop {
            let line = lines.get(index).ok_or_else(|| {
                error(
                    keyword_line,
                    format!("block for {species:?} has no data table"),
                )
            })?;
            index += 1;
            if is_dashed(line) {
                break;
            }
            let trimmed = line.trim();
            if let Some(rest) = strip_prefix_ignore_case(trimmed, "PROCESS:") {
                name = Some(rest.trim().to_string());
            } else if let Some(rest) = strip_prefix_ignore_case(trimmed, "COMMENT:") {
                comments.push(rest.trim().to_string());
            }
        }

        let mut rows: Vec<(usize, Vec<f64>)> = Vec::new();
        loop {
            let line = lines.get(index).ok_or_else(|| {
                error(
                    keyword_line,
                    format!("data table for {species:?} is not closed by a dashed line"),
                )
            })?;
            index += 1;
            if is_dashed(line) {
                break;
            }
            let trimmed = line.trim();
            if trimmed.is_empty() {
                continue;
            }
            let values: Result<Vec<f64>, _> = trimmed
                .split_whitespace()
                .map(|token| token.parse::<f64>())
                .collect();
            match values {
                Ok(values) if values.len() >= 2 => rows.push((index, values)),
                _ => {
                    return Err(error(
                        index,
                        format!("cannot read {trimmed:?} as a table row"),
                    ));
                }
            }
        }
        if rows.is_empty() {
            return Err(error(
                keyword_line,
                format!("data table for {species:?} is empty"),
            ));
        }
        let with_mt = rows[0].1.len() >= 3;
        if let Some((line, _)) = rows.iter().find(|(_, row)| (row.len() >= 3) != with_mt) {
            return Err(error(
                *line,
                "table rows must all have two or all have three columns",
            ));
        }
        let energy: Vec<f64> = rows.iter().map(|(_, row)| row[0]).collect();
        let sigma: Vec<f64> = rows.iter().map(|(_, row)| row[1]).collect();
        let table =
            Table::new(energy.clone(), sigma).map_err(|message| error(keyword_line, message))?;
        let momentum_transfer = if with_mt {
            let values = rows.iter().map(|(_, row)| row[2]).collect();
            Some(Table::new(energy, values).map_err(|message| error(keyword_line, message))?)
        } else {
            None
        };
        let section = CrossSection {
            kind,
            name: name
                .unwrap_or_else(|| format!("{species} {}", kind.keyword().to_ascii_lowercase())),
            species,
            threshold_ev,
            mass_ratio,
            weight_ratio,
            lower_state: states[0],
            upper_state: states[1],
            table,
            momentum_transfer,
            comment: comments.join("\n"),
        };
        section
            .validate()
            .map_err(|message| error(keyword_line, message))?;
        sections.push(section);
    }
    Ok(sections)
}

#[cfg(test)]
mod tests {
    use super::*;

    const SAMPLE: &str = "\u{feff}header text\r
ELASTIC\r
Ar\r
 1.371000e-5\r
PROCESS: E + Ar -> E + Ar, Elastic\r
-----\r
 0.0 7.5e-20\r
 1.0 1.0e-20\r
-----\r
EXCITATION\r
HF(J=0) -> HF(J=1)\r
 5.126e-3  3.0\r
PROCESS: rot 0-1\r
COMMENT: two numbers on line 3\r
-----\r
 5.126e-3 0.0\r
 1.0 2.0e-19\r
-----\r
EXCITATION\r
HF <-> HF(v1)\r
 4.912e-1\r
-----\r
 4.912e-1 0.0 0.0\r
 2.0 1.0e-20 4.0e-21\r
-----\r
ROTATION\r
HF\r
 0.0 1.0\r
 5.126e-3 3.0\r
-----\r
 5.126e-3 0.0\r
 1.0 2.0e-19\r
-----\r
ATTACHMENT\r
HF -> H + F^-\r
-----\r
 2.5 0.0\r
 3.0 1.0e-23\r
-----\r
";

    #[test]
    fn parses_all_block_kinds() {
        let sections = parse_str(SAMPLE).unwrap();
        assert_eq!(sections.len(), 5);
        assert_eq!(sections[0].kind, Kind::Elastic);
        assert_eq!(sections[0].mass_ratio, Some(1.371e-5));
        assert_eq!(sections[0].name, "E + Ar -> E + Ar, Elastic");
        // 3行目の2数（boltzpmp 0.1.3ではしきい値0になっていた）
        assert_eq!(sections[1].threshold_ev, 5.126e-3);
        assert_eq!(sections[1].weight_ratio, Some(3.0));
        assert_eq!(sections[1].target(), "HF(J=0)");
        assert_eq!(sections[1].product(), Some("HF(J=1)"));
        assert!(!sections[1].is_reversible());
        assert!(sections[2].is_reversible());
        assert_eq!(
            sections[2].momentum_transfer.as_ref().unwrap().values,
            vec![0.0, 4.0e-21]
        );
        assert_eq!(sections[3].kind, Kind::Rotation);
        assert_eq!(sections[3].threshold_ev, 5.126e-3);
        assert_eq!(sections[3].upper_state.unwrap().weight, 3.0);
        assert_eq!(sections[4].kind, Kind::Attachment);
        assert_eq!(sections[4].name, "HF -> H + F^- attachment");
    }

    #[test]
    fn threshold_masks_values() {
        let sections = parse_str(SAMPLE).unwrap();
        assert_eq!(sections[1].sigma_at(5.0e-3), 0.0);
        assert!(sections[1].sigma_at(0.5) > 0.0);
        assert_eq!(sections[1].sigma_at(10.0), 2.0e-19);
    }

    #[test]
    fn reports_malformed_input_with_line_numbers() {
        let missing_threshold = "EXCITATION\nAr\nPROCESS: x\n-----\n 1 0\n-----\n";
        let err = parse_str(missing_threshold).unwrap_err();
        assert_eq!(err.line, 3);
        let bad_row = "ATTACHMENT\nX\n-----\n 1 0\n 2 abc\n-----\n";
        assert_eq!(parse_str(bad_row).unwrap_err().line, 5);
        let unclosed = "ATTACHMENT\nX\n-----\n 1 0\n";
        assert!(parse_str(unclosed).is_err());
        let decreasing = "ATTACHMENT\nX\n-----\n 2 0\n 1 0\n-----\n";
        assert!(parse_str(decreasing).is_err());
        let mixed = "ATTACHMENT\nX\n-----\n 1 0 0\n 2 0\n-----\n";
        assert!(parse_str(mixed).is_err());
    }

    #[test]
    fn elastic_mass_ratio_line_is_optional() {
        let text = "ELASTIC\nAr\nPARAM.: none\n-----\n 0 1e-20\n-----\n";
        let sections = parse_str(text).unwrap();
        assert_eq!(sections[0].mass_ratio, None);
    }
}
