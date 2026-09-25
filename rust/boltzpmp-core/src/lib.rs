//! Pythonに依存しないboltzpmp数値計算コア。
//!
//! LXCat断面積の読み込み、混合気体と励起準位の占有、メッシュ上の衝突過程の組み立て、
//! プロパゲータ法によるDC・RF時間発展と出力計算までを担う。

pub mod anisotropy;
mod constants;
pub mod interp;
pub mod lxcat;
mod mesh;
pub mod mixture;
mod operators;
mod output;
pub mod processes;
mod solver;

pub use constants::{AMU, E_CHARGE, K_B, M_E, TOWNSEND, kelvin_to_ev, speed_from_ev};
pub use lxcat::{CrossSection, Kind, LevelState, ParseError, Table};
pub use mesh::{VelocityMesh, graded_edges};
pub use mixture::{Gas, Mixture, Populations};
pub use operators::{
    AdvectionOperator, AdvectionScheme, CollisionOperator, ProcessKind, ProcessSpec,
};
pub use output::SwarmScalars;
pub use processes::ModelOptions;
pub use solver::{CoreSolver, DcOptions, DcResult, RfOptions, RfResult, SolverError};
