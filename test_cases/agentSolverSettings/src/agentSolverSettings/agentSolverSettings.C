/*---------------------------------------------------------------------------*\
  =========                 |
  \\      /  F ield         | OpenFOAM: The Open Source CFD Toolbox
   \\    /   O peration     |
    \\  /    A nd           | www.openfoam.com
     \\/     M anipulation  |
-------------------------------------------------------------------------------
    Copyright (C) 2015-2016 OpenFOAM Foundation
    Copyright (C) 2015-2021 OpenCFD Ltd.
-------------------------------------------------------------------------------
License
    This file is part of OpenFOAM.

    OpenFOAM is free software: you can redistribute it and/or modify it
    under the terms of the GNU General Public License as published by
    the Free Software Foundation, either version 3 of the License, or
    (at your option) any later version.

    OpenFOAM is distributed in the hope that it will be useful, but WITHOUT
    ANY WARRANTY; without even the implied warranty of MERCHANTABILITY or
    FITNESS FOR A PARTICULAR PURPOSE.  See the GNU General Public License
    for more details.

    You should have received a copy of the GNU General Public License
    along with OpenFOAM.  If not, see <http://www.gnu.org/licenses/>.

\*---------------------------------------------------------------------------*/

#include "agentSolverSettings.H"
#include "addToRunTimeSelectionTable.H"

// * * * * * * * * * * * * * * Static Data Members * * * * * * * * * * * * * //

namespace Foam
{
namespace functionObjects
{
    defineTypeNameAndDebug(agentSolverSettings, 0);
    addToRunTimeSelectionTable(functionObject, agentSolverSettings, dictionary);
}
}


// * * * * * * * * * * * * * Protected Member Functions  * * * * * * * * * * //

void Foam::functionObjects::agentSolverSettings::writeFileHeader(Ostream& os)
{
    if (!fieldSet_.updateSelection())
    {
        return;
    }

    if (writtenHeader_)
    {
        writeBreak(file());
    }
    else
    {
        writeHeader(os, "Solver information");
    }

    writeCommented(os, "Time");

    for (const word& fieldName : fieldSet_.selectionNames())
    {
        writeFileHeader<scalar>(os, fieldName);
        writeFileHeader<vector>(os, fieldName);
        writeFileHeader<sphericalTensor>(os, fieldName);
        writeFileHeader<symmTensor>(os, fieldName);
        writeFileHeader<tensor>(os, fieldName);
    }

    os << endl;

    writtenHeader_ = true;
}


void Foam::functionObjects::agentSolverSettings::createResidualField
(
    const word& fieldName
)
{
    if (!writeResidualFields_)
    {
        return;
    }

    const word residualName
    (
        IOobject::scopedName("initialResidual", fieldName)
    );

    if (!mesh_.foundObject<IOField<scalar>>(residualName))
    {
        auto* fieldPtr =
            new IOField<scalar>
            (
                IOobject
                (
                    residualName,
                    mesh_.time().timeName(),
                    mesh_,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE
                ),
                Field<scalar>(mesh_.nCells(), Zero)
            );

        fieldPtr->store();

        residualFieldNames_.insert(residualName);
    }
}


// * * * * * * * * * * * * * * * * Constructors  * * * * * * * * * * * * * * //

Foam::functionObjects::agentSolverSettings::agentSolverSettings
(
    const word& name,
    const Time& runTime,
    const dictionary& dict
)
:
    fvMeshFunctionObject(name, runTime, dict),
    writeFile(obr_, name, typeName, dict),
    fieldSet_(mesh_),
    residualFieldNames_(),
    writeResidualFields_(false),
    initialised_(false)
{
    read(dict);
}


// * * * * * * * * * * * * * * * Member Functions  * * * * * * * * * * * * * //

bool Foam::functionObjects::agentSolverSettings::read(const dictionary& dict)
{
    if (fvMeshFunctionObject::read(dict))
    {
        initialised_ = false;

        fieldSet_.read(dict);

        writeResidualFields_ = dict.getOrDefault("writeResidualFields", false);

        residualFieldNames_.clear();

        return true;
    }

    return false;
}


bool Foam::functionObjects::agentSolverSettings::execute()
{
    // Note: delaying initialisation until after first iteration so that
    // we can find wildcard fields
    if (!initialised_)
    {
        writeFileHeader(file());

        if (writeResidualFields_)
        {
            for (const word& fieldName : fieldSet_.selectionNames())
            {
                initialiseResidualField<scalar>(fieldName);
                initialiseResidualField<vector>(fieldName);
                initialiseResidualField<sphericalTensor>(fieldName);
                initialiseResidualField<symmTensor>(fieldName);
                initialiseResidualField<tensor>(fieldName);
            }
        }

        initialised_ = true;
    }

    writeCurrentTime(file());

    for (const word& fieldName : fieldSet_.selectionNames())
    {
        updateagentSolverSettings<scalar>(fieldName);
        updateagentSolverSettings<vector>(fieldName);
        updateagentSolverSettings<sphericalTensor>(fieldName);
        updateagentSolverSettings<symmTensor>(fieldName);
        updateagentSolverSettings<tensor>(fieldName);
    }

    file() << endl;

    return true;
}


bool Foam::functionObjects::agentSolverSettings::write()
{
    for (const word& residualName : residualFieldNames_)
    {
        const auto* residualPtr =
            mesh_.findObject<IOField<scalar>>(residualName);

        if (residualPtr)
        {
            volScalarField residual
            (
                IOobject
                (
                    residualName,
                    mesh_.time().timeName(),
                    mesh_,
                    IOobject::NO_READ,
                    IOobject::NO_WRITE,
                    false
                ),
                mesh_,
                dimensionedScalar(dimless, Zero),
                zeroGradientFvPatchField<scalar>::typeName
            );

            residual.primitiveFieldRef() = *residualPtr;
            residual.correctBoundaryConditions();

            residual.write();
        }
    }

    return true;
}


// ************************************************************************* //
