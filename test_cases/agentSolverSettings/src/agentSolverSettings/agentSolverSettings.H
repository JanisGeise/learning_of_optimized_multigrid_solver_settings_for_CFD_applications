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

Class
    Foam::functionObjects::agentSolverSettings

Group
    grpUtilitiesFunctionObjects

Description
    Writes solver information for a list of user-specified fields.

    Information written to file includes:
    - residual fields
    - solver type
    - initial residual
    - final residual
    - number of solver iterations
    - convergence flag

    Operands:
    \table
      Operand      | Type                | Location
      input        | -                   | -
      output file  | dat                 <!--
               --> | postProcessing/\<FO\>/\<time\>/\<file\>(s)
      output field | volScalarField(s)   <!--
               --> | \<time\>/initialResiduals:\<outField\>(s)
    \endtable

Usage
    Minimal example by using \c system/controlDict.functions:
    \verbatim
    agentSolverSettings1
    {
        // Mandatory entries (unmodifiable)
        type            agentSolverSettings;
        libs            (utilityFunctionObjects);

        // Mandatory entries (runtime modifiable)
        fields          (<field1> <field2> ... <fieldN>);

        // Optional entries (runtime modifiable)
        writeResidualFields true;

        // Inherited entries
        ...
    }
    \endverbatim

    where the entries mean:
    \table
      Property     | Description                          | Type | Reqd | Deflt
      type         | Type name: agentSolverSettings                | word | yes  | -
      libs         | Library name: utilityFunctionObjects | word | yes  | -
      fields       | Names of operand fields          | wordList | yes  | -
      writeResidualFields | Flag to write the initial-residual fields <!--
                   -->                                    | bool | no   | false
    \endtable

    The inherited entries are elaborated in:
     - \link functionObject.H \endlink
     - \link writeFile.H \endlink

See also
  - Foam::functionObject
  - Foam::functionObjects::fvMeshFunctionObject
  - Foam::functionObjects::writeFile

SourceFiles
    agentSolverSettings.C
    agentSolverSettingsTemplates.C

\*---------------------------------------------------------------------------*/

#ifndef functionObjects_agentSolverSettings_H
#define functionObjects_agentSolverSettings_H

#include <torch/script.h>
#include "fvMeshFunctionObject.H"
#include "writeFile.H"
#include "solverFieldSelection.H"
// #include "pimpleControl.H"

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

namespace Foam
{
namespace functionObjects
{

/*---------------------------------------------------------------------------*\
                          Class agentSolverSettings Declaration
\*---------------------------------------------------------------------------*/

class agentSolverSettings
:
    public fvMeshFunctionObject,
    // public pimpleControl,
    public writeFile
{
protected:

    // Protected Data

        //- Names of operand fields
        solverFieldSelection fieldSet_;

        //- Names of (result) residual fields
        wordHashSet residualFieldNames_;

        //- Initialisation flag
        bool initialised_;

        // initialize the variables for storing the properties of the residuals
        int pimple_iter_ = 0;
        double maxSolverIter_ = 0;
        double sumSolverIter_ = 0;
        double ratio_iter_ = 0.0;
        scalar avgAbsConRate_ = 0.0;
        scalar initialResidual_ = 0.0;
        scalar maxAbsConRate_ = 0.0;
        scalar minAbsConRate_ = 42.0;
        torch::Tensor convergenceRate_;
        word solverName_;

        // initialize vector for storing the actions ('interpolateCorrection' and 'smoother')
        std::vector<int> action_ = {0, 0};

        // the following parameters are taken from drlfoam
        //- training or evaluation mode
        bool train_;

        //- name of the PyTorch model
        word policy_name_;

        //- PyTorch model
        torch::jit::script::Module policy_;

        //- seed for random sampling
        int seed_;

        //- random number generator
        std::mt19937 gen_;

    // Protected Member Functions

        //- Output file header information
        void writeFileHeader(Ostream& os, const word& fieldName);

        //- Calculate the solver information
        void computeResidualProperties(const word& fieldName);

        //- predict the solver settings based on the residuals of the last time step, adopted from drlfoam
        void predictSettings();

        //- modify the solver settings for GAMG based on the policy output
        void modifySolverSettingsDict(const word& fieldName);

        //- write the new solver settings to fvSolution
        void writeFvSolutionFile(const word& solverSettings, const word& fieldName, const int nParameters) const;

        //- write the properties of the residuals to file
        void writeResidualsToFile(Ostream& os, const word& fieldName);

        //- save the trajectory to file, taken from drlfoam, https://github.com/OFDataCommittee/drlfoam
        // void saveTrajectory(scalar prob_out) const;
        void saveTrajectory(torch::Tensor prob_out) const;

    // Generated Methods

        //- No copy construct
        agentSolverSettings(const agentSolverSettings&) = delete;

        //- No copy assignment
        void operator=(const agentSolverSettings&) = delete;


public:

    //- Runtime type information
    TypeName("agentSolverSettings");


    // Constructors

        //- Construct from Time and dictionary
        agentSolverSettings
        (
            const word& name,
            const Time& runTime,
            const dictionary& dict
        );


    //- Destructor
    virtual ~agentSolverSettings() = default;


    // Member Functions

        //- Read agentSolverSettings settings
        virtual bool read(const dictionary&);

        //- Execute agentSolverSettings
        virtual bool execute();

        //- Write agentSolverSettings results
        virtual bool write();
};


// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

} // End namespace functionObjects
} // End namespace Foam

// * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * * //

#endif

// ************************************************************************* //
