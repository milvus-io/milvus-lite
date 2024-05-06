
// Generated from Plan.g4 by ANTLR 4.13.1

#pragma once


#include "antlr4-runtime.h"
#include "PlanParser.h"



/**
 * This class defines an abstract visitor for a parse tree
 * produced by PlanParser.
 */
class  PlanVisitor : public antlr4::tree::AbstractParseTreeVisitor {
public:

  /**
   * Visit parse trees produced by PlanParser.
   */
    virtual std::any visitJSONIdentifier(PlanParser::JSONIdentifierContext *context) = 0;

    virtual std::any visitParens(PlanParser::ParensContext *context) = 0;

    virtual std::any visitString(PlanParser::StringContext *context) = 0;

    virtual std::any visitFloating(PlanParser::FloatingContext *context) = 0;

    virtual std::any visitJSONContainsAll(PlanParser::JSONContainsAllContext *context) = 0;

    virtual std::any visitLogicalOr(PlanParser::LogicalOrContext *context) = 0;

    virtual std::any visitMulDivMod(PlanParser::MulDivModContext *context) = 0;

    virtual std::any visitIdentifier(PlanParser::IdentifierContext *context) = 0;

    virtual std::any visitLike(PlanParser::LikeContext *context) = 0;

    virtual std::any visitLogicalAnd(PlanParser::LogicalAndContext *context) = 0;

    virtual std::any visitEquality(PlanParser::EqualityContext *context) = 0;

    virtual std::any visitBoolean(PlanParser::BooleanContext *context) = 0;

    virtual std::any visitShift(PlanParser::ShiftContext *context) = 0;

    virtual std::any visitReverseRange(PlanParser::ReverseRangeContext *context) = 0;

    virtual std::any visitBitOr(PlanParser::BitOrContext *context) = 0;

    virtual std::any visitAddSub(PlanParser::AddSubContext *context) = 0;

    virtual std::any visitRelational(PlanParser::RelationalContext *context) = 0;

    virtual std::any visitArrayLength(PlanParser::ArrayLengthContext *context) = 0;

    virtual std::any visitTerm(PlanParser::TermContext *context) = 0;

    virtual std::any visitJSONContains(PlanParser::JSONContainsContext *context) = 0;

    virtual std::any visitRange(PlanParser::RangeContext *context) = 0;

    virtual std::any visitUnary(PlanParser::UnaryContext *context) = 0;

    virtual std::any visitInteger(PlanParser::IntegerContext *context) = 0;

    virtual std::any visitArray(PlanParser::ArrayContext *context) = 0;

    virtual std::any visitJSONContainsAny(PlanParser::JSONContainsAnyContext *context) = 0;

    virtual std::any visitBitXor(PlanParser::BitXorContext *context) = 0;

    virtual std::any visitExists(PlanParser::ExistsContext *context) = 0;

    virtual std::any visitBitAnd(PlanParser::BitAndContext *context) = 0;

    virtual std::any visitEmptyTerm(PlanParser::EmptyTermContext *context) = 0;

    virtual std::any visitPower(PlanParser::PowerContext *context) = 0;


};

