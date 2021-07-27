using SMACOF
using Documenter

DocMeta.setdocmeta!(SMACOF, :DocTestSetup, :(using SMACOF); recursive=true)

makedocs(;
    modules=[SMACOF],
    authors="Alain Chau <phchau@ucsd.edu> and contributors",
    repo="https://github.com/alainchau/SMACOF.jl/blob/{commit}{path}#{line}",
    sitename="SMACOF.jl",
    format=Documenter.HTML(;
        prettyurls=get(ENV, "CI", "false") == "true",
        canonical="https://alainchau.github.io/SMACOF.jl",
        assets=String[],
    ),
    pages=[
        "Home" => "index.md",
    ],
)

deploydocs(;
    repo="github.com/alainchau/SMACOF.jl",
)
