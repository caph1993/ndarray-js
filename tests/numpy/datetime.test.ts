import { numpy_cases } from "../helpers/cases";
import { np } from "../../src/index";

numpy_cases("busday functions (dates as strings)", [
  ["np.is_busday(['2011-07-01', '2011-07-02', '2011-07-03'])", "np.is_busday(['2011-07-01', '2011-07-02', '2011-07-03'])"],
  ["np.is_busday('2011-07-04', holidays=['2011-07-04'])", "np.is_busday('2011-07-04', holidays=['2011-07-04'])"],
  ["np.is_busday(['2011-07-01', '2011-07-02'], weekmask='Sat Sun')", "np.is_busday(['2011-07-01', '2011-07-02'], weekmask='Sat Sun')"],
  ["np.is_busday(['2011-07-01', '2011-07-02'], weekmask=[0,0,0,0,0,1,1])", "np.is_busday(['2011-07-01', '2011-07-02'], weekmask=[0,0,0,0,0,1,1])"],
  ["np.busday_count('2011-01-01', '2011-01-10')", "np.busday_count('2011-01-01', '2011-01-10')"],
  ["np.busday_count('2011-01-10', '2011-01-01')", "np.busday_count('2011-01-10', '2011-01-01')"],
  ["np.busday_count('2011-01-01', '2011-01-01')", "np.busday_count('2011-01-01', '2011-01-01')"],
  ["np.busday_count('2011-01-01', '2012-06-01')", "np.busday_count('2011-01-01', '2012-06-01')"],
  ["np.busday_count('2011-01-01', '2011-01-31', holidays=['2011-01-17', '2011-01-03'])", "np.busday_count('2011-01-01', '2011-01-31', holidays=['2011-01-17', '2011-01-03'])"],
  ["np.busday_count(['2011-01-01', '2011-02-01'], '2011-03-01')", "np.busday_count(['2011-01-01', '2011-02-01'], '2011-03-01')"],
  ["np.busday_count('2011-01-01', '2011-01-20', weekmask='Mon Wed')", "np.busday_count('2011-01-01', '2011-01-20', weekmask='Mon Wed')"],
  ["np.busday_offset('2011-03-20', 0, roll='forward')", "str(np.busday_offset('2011-03-20', 0, roll='forward'))"],
  ["np.busday_offset('2011-03-20', 1, roll='forward')", "str(np.busday_offset('2011-03-20', 1, roll='forward'))"],
  ["np.busday_offset('2011-03-20', -1, roll='forward')", "str(np.busday_offset('2011-03-20', -1, roll='forward'))"],
  ["np.busday_offset('2011-03-20', 0, roll='backward')", "str(np.busday_offset('2011-03-20', 0, roll='backward'))"],
  ["np.busday_offset('2011-03-20', 2, roll='modifiedpreceding')", "str(np.busday_offset('2011-03-20', 2, roll='modifiedpreceding'))"],
  ["np.busday_offset('2011-10-30', 0, roll='modifiedfollowing')", "str(np.busday_offset('2011-10-30', 0, roll='modifiedfollowing'))"],
  ["np.busday_offset('2011-07-01', 100)", "str(np.busday_offset('2011-07-01', 100))"],
  ["np.busday_offset('2011-07-01', -37)", "str(np.busday_offset('2011-07-01', -37))"],
  ["np.busday_offset('2011-07-01', 10, holidays=['2011-07-04'])", "str(np.busday_offset('2011-07-01', 10, holidays=['2011-07-04']))"],
  ["np.busday_offset(['2011-07-01', '2011-07-08'], [1, 2]).tolist()", "np.busday_offset(['2011-07-01', '2011-07-08'], [1, 2]).astype(str).tolist()"],
  ["!np.busday_offset('2011-03-20', 1)", "!np.busday_offset('2011-03-20', 1)"],
  ["np.datetime_as_string('2011-07-01')", "str(np.datetime_as_string(np.datetime64('2011-07-01')))"],
  ["np.datetime_as_string('2011-07-01', 'M')", "str(np.datetime_as_string(np.datetime64('2011-07-01'), unit='M'))"],
  ["np.datetime_as_string('2011-07-01', 's')", "str(np.datetime_as_string(np.datetime64('2011-07-01'), unit='s'))"],
  ["np.datetime_data('datetime64[D]')", "list(np.datetime_data(np.dtype('M8[D]')))"],
  ["np.isnat('NaT')", "np.isnat(np.datetime64('NaT'))"],
  ["np.isnat(['2011-01-01', 'NaT'])", "np.isnat(np.array(['2011-01-01', 'NaT'], dtype='M8[D]'))"],
  ["np.busdaycalendar(weekmask='Mon Tue').weekmask", "np.busdaycalendar(weekmask='Mon Tue').weekmask"],
]);

import { describe, it, expect } from "vitest";
describe("busdaycalendar", () => {
  it("sorts and deduplicates holidays", () => {
    const cal = (np as any).busdaycalendar('1111100', ['2011-07-01', '2011-01-01', '2011-07-01']);
    expect(cal.holidays.tolist()).toEqual(['2011-01-01', '2011-07-01']);
    expect((np as any).busday_count('2011-06-27', '2011-07-04', null, null, cal)).toBe(4);
  });
});
