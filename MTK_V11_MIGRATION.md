# ModelingToolkit v11 Migration Report

## Summary
Migrated CirculatorySystemModels.jl from ModelingToolkit v10 to v11, converting all `@mtkmodel` macro-based definitions to function-based `System` constructors.

## Breaking Changes in MTK v11 Addressed

### 1. `@mtkmodel` Macro Removed
MTK v11 removes the `@mtkmodel` macro in favor of function-based component definitions using the `System` constructor with `compose` and `extend`.

**Before (MTK v10):**
```julia
@mtkmodel Resistor begin
    @extend OnePort()
    @parameters begin
        R = 1.0
    end
    @equations begin
        Δp ~ -q * R
    end
end
```

**After (MTK v11):**
```julia
function Resistor(; name, R=1.0)
    @named oneport = OnePort()
    @unpack q, in, out = oneport
    ps = @parameters R = R
    eqs = [out.p - in.p ~ -q * R]
    extend(System(eqs, t, [], ps; name), oneport)
end
```

### 2. Connector Type Declaration
MTK v11 requires explicit connector type declaration. Added `connector_type=Flow` to Pin:

```julia
function Pin(; name)
    vars = @variables begin
        p(t)
        q(t), [connect = Flow]
    end
    System(Equation[], t, vars, []; name, connector_type=Flow)
end
```

### 3. Independent Variable Access
Changed from `@parameters t` to using MTK's built-in `t_nounits` and `D_nounits`:

```julia
using ModelingToolkit: t_nounits as t, D_nounits as D, Flow
```

## Key Technical Fix: Duplicate Equation Issue

The original migration caused an "ExtraEquationsSystemException: 23 equations for 22 unknowns" error.

**Root Cause:** OnePort defined `Δp ~ out.p - in.p`, and extending components like Resistor added `Δp ~ -q * R`. With `extend()`, both equations were kept, creating overdetermination.

**Solution:** Removed `Δp` from OnePort entirely. Components now either:
- Use `out.p - in.p` directly in equations (simple components)
- Define `Δp` as a local state variable with its definition (complex components needing Δp in multiple places)

## Files Modified

| File | Changes |
|------|---------|
| `src/CirculatorySystemModels.jl` | Converted 30+ components from macro to function-based |
| `test/runtests.jl` | Updated "Shi Model Complex" test to use function-based approach |
| `Project.toml` | Updated ModelingToolkit compat from "10" to "11" |

## Components Converted

### Base Components
- `Pin` - Added `connector_type=Flow`
- `Ground` - Function-based with `compose`
- `OnePort` - Removed `Δp` variable and equation

### Simple OnePort Extensions (use `out.p - in.p` directly)
- `Resistor`, `QResistor`, `PoiseuilleResistor`
- `Inductance`
- `ConstantPressure`, `ConstantFlow`
- `DrivenPressure`, `DrivenFlow`

### Components with Local Δp State
- `Capacitor` - needs Δp for `D(Δp) ~ -q / C`
- `ResistorDiode`, `OrificeValve` - Δp in conditional expressions
- `ShiValve` - Δp in multiple force equations
- `MynardValve_SemiLunar`, `MynardValve_Atrioventricular` - Δp scaling

### Windkessel Models (removed unused Δp from unpack)
- `WK3`, `WK3E`, `WK4_S`, `WK4_SE`, `WK4_P`, `WK4_PE`, `WK5`, `WK5E`

### Composite Components
- `Compliance`, `Elastance`, `VariableElastance`
- `DHChamber`, `ShiChamber`, `ShiAtrium`
- `ShiHeart`, `CR`, `CRL`, `RRCR`
- `ShiSystemicLoop`, `ShiPulmonaryLoop`

## Test Results

All 4 test sets pass:
- **Shi Model in V**: 5/5 passed
- **Shi Model in P**: 5/5 passed
- **Shi Model Complex**: 5/5 passed
- **Bjørdalsbakke**: 4/4 passed
